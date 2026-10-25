use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::audio_source::AudioSource;
use crate::inference::coreml::{CachedInputShape, SharedCoreMlModel};

use super::gpu::{PreparedChunk, TaggedPrepared};
use super::{PipelineError, backend_error, chunk_audio_raw, write_speaker_mask_to_slice};

impl PrepScratch {
    pub(super) fn new(window_samples: usize) -> Self {
        Self {
            fbank_30s_buf: vec![0.0f32; 480_000],
            waveform_10s_buf: vec![0.0f32; window_samples],
            fbank_30s_shape: CachedInputShape::new("waveform", &[1, 1, 480_000]),
            fbank_10s_shape: CachedInputShape::new("waveform", &[1, 1, window_samples]),
        }
    }
}

impl ChunkPrep {
    pub(super) fn chunk_win_capacity(&self) -> usize {
        self.max_active / self.num_speakers
    }

    fn compute_chunk_fbank(
        &self,
        global_start: usize,
        num_windows: usize,
        audio: &[f32],
        scratch: &mut PrepScratch,
    ) -> Result<Vec<f32>, PipelineError> {
        let chunk_audio_start = global_start * self.step_samples;
        let chunk_audio_len = self.window_samples + (num_windows - 1) * self.step_samples;
        let chunk_audio_end = (chunk_audio_start + chunk_audio_len).min(audio.len());
        let chunk_audio = &audio[chunk_audio_start..chunk_audio_end];

        let mut fbank = vec![0.0f32; self.largest_fbank_frames * 80];

        if chunk_audio.len() <= 480_000 {
            if let Some(fbank_model) = &self.fbank_30s {
                scratch.fbank_30s_buf[..chunk_audio.len()].copy_from_slice(chunk_audio);
                scratch.fbank_30s_buf[chunk_audio.len()..].fill(0.0);
                let (data, out_shape) = fbank_model
                    .predict_cached(&[(&scratch.fbank_30s_shape, &*scratch.fbank_30s_buf)])
                    .map_err(|error| backend_error("chunk fbank 30s prediction failed", error))?;
                let copy_frames = out_shape[1].min(self.largest_fbank_frames);
                for row_idx in 0..copy_frames {
                    let offset = row_idx * 80;
                    fbank[offset..offset + 80].copy_from_slice(&data[offset..offset + 80]);
                }
            }
        } else if let Some(fbank_model) = &self.fbank_10s {
            let mut fbank_offset = 0usize;
            let mut audio_offset = 0usize;
            while fbank_offset < self.largest_fbank_frames && audio_offset < chunk_audio.len() {
                let segment_end = (audio_offset + self.window_samples).min(chunk_audio.len());
                let segment_len = segment_end - audio_offset;
                scratch.waveform_10s_buf[..segment_len]
                    .copy_from_slice(&chunk_audio[audio_offset..segment_end]);
                if segment_len < self.window_samples {
                    scratch.waveform_10s_buf[segment_len..].fill(0.0);
                }
                let (data, out_shape) = fbank_model
                    .predict_cached(&[(&scratch.fbank_10s_shape, &*scratch.waveform_10s_buf)])
                    .map_err(|error| backend_error("chunk fbank 10s prediction failed", error))?;
                let copy = out_shape[1].min(self.largest_fbank_frames - fbank_offset);
                for row_idx in 0..copy {
                    let src = row_idx * 80;
                    let dst = (fbank_offset + row_idx) * 80;
                    fbank[dst..dst + 80].copy_from_slice(&data[src..src + 80]);
                }
                fbank_offset += 998;
                audio_offset += self.window_samples;
            }
        }

        Ok(fbank)
    }

    fn collect_chunk_masks(
        &self,
        global_start: usize,
        decoded_chunk: &[ndarray::Array2<f32>],
        audio: &[f32],
    ) -> (Vec<f32>, Vec<(usize, usize)>) {
        let mut masks = vec![0.0f32; self.largest_num_masks * 589];
        let mut active = Vec::with_capacity(self.max_active);

        for (local_idx, decoded) in decoded_chunk.iter().enumerate() {
            let global_idx = global_start + local_idx;
            let win_audio =
                chunk_audio_raw(audio, self.step_samples, self.window_samples, global_idx);
            for speaker_idx in 0..self.num_speakers {
                let mask_idx = local_idx * self.num_speakers + speaker_idx;
                if mask_idx >= self.largest_num_masks {
                    break;
                }
                let dest = &mut masks[mask_idx * 589..mask_idx * 589 + 589];
                if write_speaker_mask_to_slice(
                    &decoded.view(),
                    speaker_idx,
                    win_audio.len(),
                    self.min_num_samples,
                    dest,
                ) {
                    active.push((local_idx, speaker_idx));
                }
            }
        }

        (masks, active)
    }

    pub(super) fn prep(
        &self,
        decoded: DecodedChunk,
        audio: &[f32],
        scratch: &mut PrepScratch,
    ) -> Result<PreparedChunk, PipelineError> {
        let fbank = self.compute_chunk_fbank(
            decoded.global_start,
            decoded.decoded_chunk.len(),
            audio,
            scratch,
        )?;
        let (masks, active) =
            self.collect_chunk_masks(decoded.global_start, &decoded.decoded_chunk, audio);

        Ok(PreparedChunk {
            global_start: decoded.global_start,
            decoded_chunk: decoded.decoded_chunk,
            fbank,
            masks,
            active,
            num_masks: self.largest_num_masks,
        })
    }

    pub(super) fn prep_source<S: AudioSource + ?Sized>(
        &self,
        decoded: DecodedChunk,
        source: &S,
        scratch: &mut PrepScratch,
    ) -> Result<PreparedChunk, PipelineError> {
        let global_start = decoded.global_start;
        let audio = read_source_chunk(
            source,
            global_start,
            decoded.decoded_chunk.len(),
            self.step_samples,
            self.window_samples,
        )?;
        let local = DecodedChunk {
            global_start: 0,
            decoded_chunk: decoded.decoded_chunk,
        };
        let mut prepared = self.prep(local, &audio, scratch)?;
        prepared.global_start = global_start;
        Ok(prepared)
    }
}

pub(super) fn read_source_chunk<S: AudioSource + ?Sized>(
    source: &S,
    global_start: usize,
    num_windows: usize,
    step_samples: usize,
    window_samples: usize,
) -> Result<Vec<f32>, PipelineError> {
    if num_windows == 0 {
        return Err(super::invariant_error("source chunk contains no windows"));
    }
    let start = global_start
        .checked_mul(step_samples)
        .ok_or_else(|| super::invariant_error("source chunk offset overflowed"))?;
    let span = window_samples
        .checked_add(
            (num_windows - 1)
                .checked_mul(step_samples)
                .ok_or_else(|| super::invariant_error("source chunk length overflowed"))?,
        )
        .ok_or_else(|| super::invariant_error("source chunk length overflowed"))?;
    let available = source
        .sample_count()
        .checked_sub(start)
        .ok_or_else(|| super::invariant_error("source chunk starts beyond the recording"))?;
    if available < window_samples {
        return Err(super::invariant_error(
            "source chunk has no complete window",
        ));
    }
    let mut audio = vec![0.0; span.min(available)];
    source
        .read_exact_samples(start, &mut audio)
        .map_err(|error| {
            PipelineError::Segmentation(
                crate::inference::segmentation::SegmentationError::AudioSource(error),
            )
        })?;
    Ok(audio)
}

pub(super) struct PrepWorker {
    pub(super) prep: ChunkPrep,
    pub(super) scratch: PrepScratch,
}

impl PrepWorker {
    pub(super) fn run(
        mut self,
        audio: &[f32],
        step_samples: usize,
        window_samples: usize,
        chunk_rx: &Receiver<DecodedChunk>,
        prep_tx: Sender<PreparedChunk>,
    ) -> Result<PrepStats, PipelineError> {
        let mut stats = PrepStats::default();

        while let Ok(decoded) = chunk_rx.recv() {
            let chunk_audio_start = decoded.global_start * step_samples;
            if chunk_audio_start + window_samples > audio.len() {
                continue;
            }
            let prep_start = std::time::Instant::now();
            let prepared = self.prep.prep(decoded, audio, &mut self.scratch)?;
            stats.fbank_us += prep_start.elapsed().as_micros() as u64;
            stats.chunks += 1;
            if prep_tx.send(prepared).is_err() {
                break;
            }
        }
        Ok(stats)
    }

    pub(super) fn run_source<S: AudioSource + ?Sized>(
        mut self,
        source: &S,
        chunk_rx: &Receiver<DecodedChunk>,
        prep_tx: Sender<PreparedChunk>,
    ) -> Result<PrepStats, PipelineError> {
        let mut stats = PrepStats::default();
        while let Ok(decoded) = chunk_rx.recv() {
            let start = decoded
                .global_start
                .checked_mul(self.prep.step_samples)
                .ok_or_else(|| super::invariant_error("source chunk offset overflowed"))?;
            if start
                .checked_add(self.prep.window_samples)
                .is_none_or(|end| end > source.sample_count())
            {
                continue;
            }
            let prep_start = std::time::Instant::now();
            let prepared = self.prep.prep_source(decoded, source, &mut self.scratch)?;
            stats.fbank_us += prep_start.elapsed().as_micros() as u64;
            stats.chunks += 1;
            if prep_tx.send(prepared).is_err() {
                break;
            }
        }
        Ok(stats)
    }
}

pub(super) struct BatchPrepWorker {
    pub(super) prep: ChunkPrep,
    pub(super) scratch: PrepScratch,
}

impl BatchPrepWorker {
    pub(super) fn run(
        mut self,
        audios: &[&[f32]],
        decoded_rx: &Receiver<TaggedDecoded>,
        prepared_tx: Sender<TaggedPrepared>,
    ) -> Result<PrepStats, PipelineError> {
        let mut stats = PrepStats::default();

        while let Ok(tagged) = decoded_rx.recv() {
            let audio = audios[tagged.file_idx];
            let decoded = DecodedChunk {
                global_start: tagged.local_start * self.prep.chunk_win_capacity(),
                decoded_chunk: tagged.decoded_chunk,
            };
            let chunk_audio_start = decoded.global_start * self.prep.step_samples;
            if chunk_audio_start + self.prep.window_samples > audio.len() {
                continue;
            }
            let prep_start = std::time::Instant::now();
            let prepared = self.prep.prep(decoded, audio, &mut self.scratch)?;
            stats.fbank_us += prep_start.elapsed().as_micros() as u64;
            stats.chunks += 1;
            if prepared_tx
                .send(TaggedPrepared {
                    file_idx: tagged.file_idx,
                    local_start: tagged.local_start,
                    prepared,
                })
                .is_err()
            {
                break;
            }
        }
        Ok(stats)
    }
}

#[derive(Clone)]
pub(super) struct ChunkPrep {
    pub(super) step_samples: usize,
    pub(super) window_samples: usize,
    pub(super) num_speakers: usize,
    pub(super) min_num_samples: usize,
    pub(super) largest_fbank_frames: usize,
    pub(super) largest_num_masks: usize,
    pub(super) max_active: usize,
    pub(super) fbank_30s: Option<Arc<SharedCoreMlModel>>,
    pub(super) fbank_10s: Option<Arc<SharedCoreMlModel>>,
}

pub(super) struct PrepScratch {
    pub(super) fbank_30s_buf: Vec<f32>,
    pub(super) waveform_10s_buf: Vec<f32>,
    pub(super) fbank_30s_shape: CachedInputShape,
    pub(super) fbank_10s_shape: CachedInputShape,
}

#[derive(Default)]
pub(super) struct PrepStats {
    pub(super) chunks: u32,
    pub(super) fbank_us: u64,
}

pub(super) struct DecodedChunk {
    pub(super) global_start: usize,
    pub(super) decoded_chunk: Vec<ndarray::Array2<f32>>,
}

pub(super) struct TaggedDecoded {
    pub(super) file_idx: usize,
    pub(super) local_start: usize,
    pub(super) decoded_chunk: Vec<ndarray::Array2<f32>>,
}

#[cfg(test)]
mod source_chunk_tests {
    use super::read_source_chunk;
    use crate::audio_source::{AudioSource, AudioSourceError};

    struct TrackingSource {
        samples: Vec<f32>,
        max_read: std::sync::atomic::AtomicUsize,
        cancel: bool,
    }

    impl AudioSource for TrackingSource {
        fn sample_count(&self) -> usize {
            self.samples.len()
        }

        fn read_exact_samples(
            &self,
            start: usize,
            destination: &mut [f32],
        ) -> Result<(), AudioSourceError> {
            self.max_read
                .fetch_max(destination.len(), std::sync::atomic::Ordering::Relaxed);
            if self.cancel {
                return Err(AudioSourceError::Cancelled);
            }
            let end = start
                .checked_add(destination.len())
                .ok_or(AudioSourceError::Overflow)?;
            destination.copy_from_slice(
                self.samples
                    .get(start..end)
                    .ok_or(AudioSourceError::InvalidRange)?,
            );
            Ok(())
        }
    }

    #[test]
    fn source_embedding_batch_reads_only_its_bounded_chunk_range() {
        let source = TrackingSource {
            samples: (0..256).map(|sample| sample as f32).collect(),
            max_read: std::sync::atomic::AtomicUsize::new(0),
            cancel: false,
        };
        let chunk = read_source_chunk(&source, 5, 64, 2, 4).unwrap();
        assert_eq!(chunk.len(), 130);
        assert_eq!(chunk.first(), Some(&10.0));
        assert_eq!(chunk.last(), Some(&139.0));
        assert_eq!(
            source.max_read.load(std::sync::atomic::Ordering::Relaxed),
            130
        );
    }

    #[test]
    fn source_embedding_batch_propagates_cancellation() {
        let source = TrackingSource {
            samples: vec![0.0; 256],
            max_read: std::sync::atomic::AtomicUsize::new(0),
            cancel: true,
        };
        assert!(matches!(
            read_source_chunk(&source, 0, 32, 2, 4),
            Err(crate::pipeline::PipelineError::Segmentation(
                crate::inference::segmentation::SegmentationError::AudioSource(
                    AudioSourceError::Cancelled
                )
            ))
        ));
    }
}
