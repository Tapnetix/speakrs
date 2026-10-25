use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::audio_source::AudioSource;
use crate::inference::coreml::{CachedInputShape, SharedCoreMlModel};

use super::orchestrate::source_chunk_has_complete_first_window;
use super::prep::{DecodedChunk, PrepScratch, TaggedDecoded};
use super::{EmbeddingModel, PipelineError, backend_error, invariant_error};

fn prepare_source_if_complete<S, F>(
    source: &S,
    decoded: DecodedChunk,
    step_samples: usize,
    window_samples: usize,
    prepare: F,
) -> Result<Option<PreparedChunk>, PipelineError>
where
    S: AudioSource + ?Sized,
    F: FnOnce(DecodedChunk, &S) -> Result<PreparedChunk, PipelineError>,
{
    if !source_chunk_has_complete_first_window(
        source.sample_count(),
        decoded.global_start,
        step_samples,
        window_samples,
    )? {
        return Ok(None);
    }
    let prepared = prepare(decoded, source)?;
    Ok(Some(prepared))
}

#[cfg(test)]
mod source_tail_tests {
    use super::{
        invariant_error, prepare_source_if_complete, source_chunk_has_complete_first_window,
    };
    use crate::audio_source::{AudioSource, AudioSourceError};
    use crate::pipeline::chunk_embedding::prep::DecodedChunk;
    use ndarray::Array2;

    struct GeometryOnlySource(usize);

    impl AudioSource for GeometryOnlySource {
        fn sample_count(&self) -> usize {
            self.0
        }

        fn read_exact_samples(
            &self,
            _start: usize,
            _destination: &mut [f32],
        ) -> Result<(), AudioSourceError> {
            unreachable!("the GPU worker must skip an incomplete source group before reading")
        }
    }

    #[test]
    fn gpu_worker_skips_tail_only_group_after_two_full_116_window_groups() {
        let capacity = 116;
        let full_windows = capacity * 2;
        let window_samples = 160_000;
        let step_samples = 15_360;
        let sample_count = 3_715_160;
        assert!(
            source_chunk_has_complete_first_window(
                sample_count,
                full_windows - 1,
                step_samples,
                window_samples,
            )
            .unwrap()
        );
        assert!(
            !source_chunk_has_complete_first_window(
                sample_count,
                full_windows,
                step_samples,
                window_samples,
            )
            .unwrap()
        );
        let source = GeometryOnlySource(sample_count);
        let decoded = DecodedChunk {
            global_start: full_windows,
            decoded_chunk: vec![Array2::zeros((1, 1))],
        };
        let mut prep_called = false;

        let prepared = prepare_source_if_complete(
            &source,
            decoded,
            step_samples,
            window_samples,
            |_decoded, _source| {
                prep_called = true;
                Err(invariant_error("source chunk has no complete window"))
            },
        )
        .unwrap();

        assert!(
            prepared.is_none(),
            "GPU worker must discard the tail-only group"
        );
        assert!(
            !prep_called,
            "GPU worker must not call prep_source for the tail-only group"
        );
    }
}

impl ChunkEmbeddingResources {
    pub(super) fn largest_session(&self) -> Result<LargestChunkSession, PipelineError> {
        let session = self
            .chunk_sessions
            .last()
            .cloned()
            .ok_or_else(|| invariant_error("missing chunk embedding session"))?;
        let (_, fbank_frames, num_masks) = self
            .chunk_lookup
            .last()
            .copied()
            .ok_or_else(|| invariant_error("missing chunk embedding session metadata"))?;

        Ok(LargestChunkSession {
            session,
            fbank_frames,
            num_masks,
        })
    }
}

pub(super) fn chunk_embedding_resources(
    emb_model: &mut EmbeddingModel,
) -> Result<Option<ChunkEmbeddingResources>, PipelineError> {
    let Some(bundle) = emb_model.prepare_chunk_resources()? else {
        return Ok(None);
    };

    let chunk_sessions = bundle
        .sessions
        .iter()
        .map(|session| ChunkSessionHandle {
            cached_fbank_shape: Arc::clone(&session.cached_fbank_shape),
            cached_masks_shape: Arc::clone(&session.cached_masks_shape),
            model: Arc::clone(&session.model),
        })
        .collect();

    let chunk_lookup = bundle
        .sessions
        .iter()
        .map(|session| (session.num_windows, session.fbank_frames, session.num_masks))
        .collect();

    Ok(Some(ChunkEmbeddingResources {
        chunk_sessions,
        chunk_lookup,
        fbank_30s: bundle.fbank_30s,
        fbank_10s: bundle.fbank_10s,
    }))
}

pub(super) struct GpuWorker {
    pub(super) model: Arc<SharedCoreMlModel>,
    pub(super) fbank_shape: Arc<CachedInputShape>,
    pub(super) masks_shape: Arc<CachedInputShape>,
    pub(super) prep: super::ChunkPrep,
    pub(super) scratch: PrepScratch,
}

impl GpuWorker {
    fn next_prepared(
        &mut self,
        audio: &[f32],
        prep_rx: &Receiver<PreparedChunk>,
        chunk_rx: &Receiver<DecodedChunk>,
        decoded_done: &mut bool,
        total_prep_us: &mut u64,
    ) -> Result<Option<PreparedChunk>, PipelineError> {
        match prep_rx.try_recv() {
            Ok(prepared) => return Ok(Some(prepared)),
            Err(crossbeam_channel::TryRecvError::Disconnected) => return Ok(None),
            Err(crossbeam_channel::TryRecvError::Empty) => {}
        }

        if *decoded_done {
            return match prep_rx.recv() {
                Ok(prepared) => Ok(Some(prepared)),
                Err(_) => Ok(None),
            };
        }

        match chunk_rx.try_recv() {
            Ok(decoded) => {
                let prep_start = std::time::Instant::now();
                let prepared = self.prep.prep(decoded, audio, &mut self.scratch)?;
                *total_prep_us += prep_start.elapsed().as_micros() as u64;
                Ok(Some(prepared))
            }
            Err(crossbeam_channel::TryRecvError::Empty) => crossbeam_channel::select! {
                recv(prep_rx) -> message => match message {
                    Ok(prepared) => Ok(Some(prepared)),
                    Err(_) => Ok(None),
                },
                recv(chunk_rx) -> message => match message {
                    Ok(decoded) => {
                        let prep_start = std::time::Instant::now();
                        let prepared = self.prep.prep(decoded, audio, &mut self.scratch)?;
                        *total_prep_us += prep_start.elapsed().as_micros() as u64;
                        Ok(Some(prepared))
                    }
                    Err(_) => {
                        *decoded_done = true;
                        match prep_rx.recv() {
                            Ok(prepared) => Ok(Some(prepared)),
                            Err(_) => Ok(None),
                        }
                    }
                },
            },
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                *decoded_done = true;
                match prep_rx.recv() {
                    Ok(prepared) => Ok(Some(prepared)),
                    Err(_) => Ok(None),
                }
            }
        }
    }

    fn next_prepared_source<S: AudioSource + ?Sized>(
        &mut self,
        source: &S,
        prep_rx: &Receiver<PreparedChunk>,
        chunk_rx: &Receiver<DecodedChunk>,
        decoded_done: &mut bool,
        total_prep_us: &mut u64,
    ) -> Result<Option<PreparedChunk>, PipelineError> {
        loop {
            match prep_rx.try_recv() {
                Ok(prepared) => return Ok(Some(prepared)),
                Err(crossbeam_channel::TryRecvError::Disconnected) => return Ok(None),
                Err(crossbeam_channel::TryRecvError::Empty) => {}
            }
            if *decoded_done {
                return Ok(prep_rx.recv().ok());
            }

            let decoded = match chunk_rx.try_recv() {
                Ok(decoded) => decoded,
                Err(crossbeam_channel::TryRecvError::Empty) => crossbeam_channel::select! {
                    recv(prep_rx) -> message => return Ok(message.ok()),
                    recv(chunk_rx) -> message => match message {
                        Ok(decoded) => decoded,
                        Err(_) => {
                            *decoded_done = true;
                            return Ok(prep_rx.recv().ok());
                        }
                    },
                },
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    *decoded_done = true;
                    return Ok(prep_rx.recv().ok());
                }
            };

            let prep_start = std::time::Instant::now();
            let prepared = prepare_source_if_complete(
                source,
                decoded,
                self.prep.step_samples,
                self.prep.window_samples,
                |decoded, source| self.prep.prep_source(decoded, source, &mut self.scratch),
            )?;
            if let Some(prepared) = prepared {
                *total_prep_us += prep_start.elapsed().as_micros() as u64;
                return Ok(Some(prepared));
            }
        }
    }

    fn predict(&self, prepared: &PreparedChunk) -> Result<(Vec<f32>, u64), PipelineError> {
        let predict_start = std::time::Instant::now();
        let (data, _) = self
            .model
            .predict_cached(&[
                (&*self.fbank_shape, &prepared.fbank),
                (&*self.masks_shape, &prepared.masks),
            ])
            .map_err(|error| backend_error("chunk embedding prediction failed", error))?;
        Ok((data, predict_start.elapsed().as_micros() as u64))
    }

    pub(super) fn run(
        mut self,
        audio: &[f32],
        prep_rx: Receiver<PreparedChunk>,
        chunk_rx: Receiver<DecodedChunk>,
        emb_tx: Sender<EmbeddedChunk>,
    ) -> Result<GpuStats, PipelineError> {
        let mut total_predict_us = 0u64;
        let mut total_prep_us = 0u64;
        let mut chunk_num = 0u32;
        let mut decoded_done = false;

        loop {
            let Some(prepared) = self.next_prepared(
                audio,
                &prep_rx,
                &chunk_rx,
                &mut decoded_done,
                &mut total_prep_us,
            )?
            else {
                break;
            };

            let (data, predict_us) = self.predict(&prepared)?;
            total_predict_us += predict_us;
            chunk_num += 1;

            if emb_tx
                .send(EmbeddedChunk {
                    global_start: prepared.global_start,
                    decoded_chunk: prepared.decoded_chunk,
                    data,
                    active: prepared.active,
                    num_masks: prepared.num_masks,
                    predict_us,
                })
                .is_err()
            {
                break;
            }
        }

        Ok(GpuStats {
            predict_us: total_predict_us,
            chunks: chunk_num,
            self_prep_us: total_prep_us,
        })
    }

    pub(super) fn run_source<S: AudioSource + ?Sized>(
        mut self,
        source: &S,
        prep_rx: Receiver<PreparedChunk>,
        chunk_rx: Receiver<DecodedChunk>,
        emb_tx: Sender<EmbeddedChunk>,
    ) -> Result<GpuStats, PipelineError> {
        let mut total_predict_us = 0u64;
        let mut total_prep_us = 0u64;
        let mut chunks = 0u32;
        let mut decoded_done = false;
        while let Some(prepared) = self.next_prepared_source(
            source,
            &prep_rx,
            &chunk_rx,
            &mut decoded_done,
            &mut total_prep_us,
        )? {
            let (data, predict_us) = self.predict(&prepared)?;
            total_predict_us += predict_us;
            chunks += 1;
            if emb_tx
                .send(EmbeddedChunk {
                    global_start: prepared.global_start,
                    decoded_chunk: prepared.decoded_chunk,
                    data,
                    active: prepared.active,
                    num_masks: prepared.num_masks,
                    predict_us,
                })
                .is_err()
            {
                break;
            }
        }
        Ok(GpuStats {
            predict_us: total_predict_us,
            chunks,
            self_prep_us: total_prep_us,
        })
    }
}

pub(super) struct BatchGpuWorker {
    pub(super) model: Arc<SharedCoreMlModel>,
    pub(super) fbank_shape: Arc<CachedInputShape>,
    pub(super) masks_shape: Arc<CachedInputShape>,
    pub(super) prep: super::ChunkPrep,
    pub(super) scratch: PrepScratch,
}

impl BatchGpuWorker {
    fn predict(&self, prepared: &PreparedChunk) -> Result<(Vec<f32>, u64), PipelineError> {
        let predict_start = std::time::Instant::now();
        let (data, _) = self
            .model
            .predict_cached(&[
                (&*self.fbank_shape, &prepared.fbank),
                (&*self.masks_shape, &prepared.masks),
            ])
            .map_err(|error| backend_error("batch chunk embedding prediction failed", error))?;
        Ok((data, predict_start.elapsed().as_micros() as u64))
    }

    pub(super) fn run(
        mut self,
        audios: &[&[f32]],
        prepared_rx: Receiver<TaggedPrepared>,
        decoded_rx: Receiver<TaggedDecoded>,
        embedded_tx: Sender<TaggedEmbedded>,
    ) -> Result<GpuStats, PipelineError> {
        let mut total_predict_us = 0u64;
        let mut total_prep_us = 0u64;
        let mut chunk_num = 0u32;
        let mut decoded_done = false;

        loop {
            let (file_idx, local_start, prepared) = match prepared_rx.try_recv() {
                Ok(tagged) => (tagged.file_idx, tagged.local_start, tagged.prepared),
                Err(crossbeam_channel::TryRecvError::Disconnected) => break,
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    if decoded_done {
                        match prepared_rx.recv() {
                            Ok(tagged) => (tagged.file_idx, tagged.local_start, tagged.prepared),
                            Err(_) => break,
                        }
                    } else {
                        match decoded_rx.try_recv() {
                            Ok(tagged) => {
                                let audio = audios[tagged.file_idx];
                                let decoded = DecodedChunk {
                                    global_start: tagged.local_start
                                        * self.prep.chunk_win_capacity(),
                                    decoded_chunk: tagged.decoded_chunk,
                                };
                                let prep_start = std::time::Instant::now();
                                let prepared = self.prep.prep(decoded, audio, &mut self.scratch)?;
                                total_prep_us += prep_start.elapsed().as_micros() as u64;
                                (tagged.file_idx, tagged.local_start, prepared)
                            }
                            Err(crossbeam_channel::TryRecvError::Empty) => {
                                crossbeam_channel::select! {
                                    recv(prepared_rx) -> message => match message {
                                        Ok(tagged) => {
                                            (tagged.file_idx, tagged.local_start, tagged.prepared)
                                        }
                                        Err(_) => break,
                                    },
                                    recv(decoded_rx) -> message => match message {
                                        Ok(tagged) => {
                                            let audio = audios[tagged.file_idx];
                                            let decoded = DecodedChunk {
                                                global_start: tagged.local_start * self.prep.chunk_win_capacity(),
                                                decoded_chunk: tagged.decoded_chunk,
                                            };
                                            let prep_start = std::time::Instant::now();
                                            let prepared = self.prep.prep(decoded, audio, &mut self.scratch)?;
                                            total_prep_us += prep_start.elapsed().as_micros() as u64;
                                            (tagged.file_idx, tagged.local_start, prepared)
                                        }
                                        Err(_) => {
                                            decoded_done = true;
                                            match prepared_rx.recv() {
                                                Ok(tagged) => (tagged.file_idx, tagged.local_start, tagged.prepared),
                                                Err(_) => break,
                                            }
                                        }
                                    },
                                }
                            }
                            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                                decoded_done = true;
                                match prepared_rx.recv() {
                                    Ok(tagged) => {
                                        (tagged.file_idx, tagged.local_start, tagged.prepared)
                                    }
                                    Err(_) => break,
                                }
                            }
                        }
                    }
                }
            };

            let (data, predict_us) = self.predict(&prepared)?;
            total_predict_us += predict_us;
            chunk_num += 1;

            if embedded_tx
                .send(TaggedEmbedded {
                    file_idx,
                    local_start,
                    embedded: EmbeddedChunk {
                        global_start: local_start * self.prep.chunk_win_capacity(),
                        decoded_chunk: prepared.decoded_chunk,
                        data,
                        active: prepared.active,
                        num_masks: prepared.num_masks,
                        predict_us,
                    },
                })
                .is_err()
            {
                break;
            }
        }

        Ok(GpuStats {
            predict_us: total_predict_us,
            chunks: chunk_num,
            self_prep_us: total_prep_us,
        })
    }
}

#[derive(Clone)]
pub(super) struct ChunkSessionHandle {
    pub(super) cached_fbank_shape: Arc<CachedInputShape>,
    pub(super) cached_masks_shape: Arc<CachedInputShape>,
    pub(super) model: Arc<SharedCoreMlModel>,
}

#[derive(Clone)]
pub(super) struct ChunkEmbeddingResources {
    pub(super) chunk_sessions: Vec<ChunkSessionHandle>,
    pub(super) chunk_lookup: Vec<(usize, usize, usize)>,
    pub(super) fbank_30s: Option<Arc<SharedCoreMlModel>>,
    pub(super) fbank_10s: Option<Arc<SharedCoreMlModel>>,
}

pub(super) struct LargestChunkSession {
    pub(super) session: ChunkSessionHandle,
    pub(super) fbank_frames: usize,
    pub(super) num_masks: usize,
}

pub(super) struct PreparedChunk {
    pub(super) global_start: usize,
    pub(super) decoded_chunk: Vec<ndarray::Array2<f32>>,
    pub(super) fbank: Vec<f32>,
    pub(super) masks: Vec<f32>,
    pub(super) active: Vec<(usize, usize)>,
    pub(super) num_masks: usize,
}

pub(super) struct EmbeddedChunk {
    pub(super) global_start: usize,
    pub(super) decoded_chunk: Vec<ndarray::Array2<f32>>,
    pub(super) data: Vec<f32>,
    pub(super) active: Vec<(usize, usize)>,
    pub(super) num_masks: usize,
    pub(super) predict_us: u64,
}

pub(super) struct TaggedPrepared {
    pub(super) file_idx: usize,
    pub(super) local_start: usize,
    pub(super) prepared: PreparedChunk,
}

pub(super) struct TaggedEmbedded {
    pub(super) file_idx: usize,
    pub(super) local_start: usize,
    pub(super) embedded: EmbeddedChunk,
}

pub(super) struct GpuStats {
    pub(super) predict_us: u64,
    pub(super) chunks: u32,
    pub(super) self_prep_us: u64,
}
