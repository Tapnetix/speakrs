use ndarray::{Array2, Array3};
use tracing::{debug, trace};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::audio_source::{AudioSource, AudioSourceError};
use crate::clustering::plda::PldaTransform;
use crate::inference::embedding::{ChunkEmbeddingSession, EmbeddingModel};
use crate::inference::segmentation::SegmentationModel;
use crate::powerset::PowersetMapping;

use super::config::PipelineConfig;
use super::post_inference::post_inference;
use super::types::{
    BatchInput, ChunkEmbeddings, ChunkLayout, ChunkSpeakerClusters, DecodedSegmentations,
    DiarizationResult, DiscreteDiarization, InferenceArtifacts, PipelineError, SpeakerCountTrack,
    chunk_audio_raw,
};
use super::write_speaker_mask_to_slice;

mod collect;
mod error;
mod gpu;
mod orchestrate;
mod prep;

use collect::{FileCollector, build_chunk_artifacts};
use error::{backend_error, invariant_error, worker_panic};
use gpu::{BatchGpuWorker, TaggedEmbedded, TaggedPrepared, chunk_embedding_resources};
use orchestrate::{
    run_pipelined, run_pipelined_source, run_sequential_chunks, run_sequential_chunks_source,
    seg_worker_count, setup_chunk_embedding, setup_chunk_embedding_for_len,
};
use prep::{BatchPrepWorker, ChunkPrep, DecodedChunk, PrepScratch, TaggedDecoded};

struct ChunkParams {
    step_samples: usize,
    window_samples: usize,
    num_speakers: usize,
    min_num_samples: usize,
    chunk_win_capacity: usize,
    total_windows: usize,
}

struct CancellableAudioSource<'a, S: AudioSource + ?Sized> {
    source: &'a S,
    cancelled: Arc<AtomicBool>,
}

impl<S: AudioSource + ?Sized> AudioSource for CancellableAudioSource<'_, S> {
    fn sample_count(&self) -> usize {
        self.source.sample_count()
    }

    fn read_exact_samples(
        &self,
        start: usize,
        destination: &mut [f32],
    ) -> Result<(), AudioSourceError> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(AudioSourceError::Cancelled);
        }
        self.source.read_exact_samples(start, destination)
    }
}

fn cancel_source_and_drain_on_error<T, E, U>(
    result: &Result<T, E>,
    receiver: &crossbeam_channel::Receiver<U>,
    cancelled: &AtomicBool,
) -> usize {
    if result.is_err() {
        cancelled.store(true, Ordering::Release);
        let mut drained = 0;
        while receiver.recv().is_ok() {
            drained += 1;
        }
        drained
    } else {
        0
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::{CancellableAudioSource, cancel_source_and_drain_on_error};
    use crate::audio_source::{AudioSource, AudioSourceError, SliceAudioSource};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    struct CountingSource {
        reads: Arc<std::sync::atomic::AtomicUsize>,
        sample_count: usize,
    }

    impl AudioSource for CountingSource {
        fn sample_count(&self) -> usize {
            self.sample_count
        }

        fn read_exact_samples(
            &self,
            start: usize,
            destination: &mut [f32],
        ) -> Result<(), AudioSourceError> {
            if start + destination.len() > self.sample_count {
                return Err(AudioSourceError::InvalidRange);
            }
            self.reads.fetch_add(1, Ordering::Relaxed);
            destination.fill(start as f32);
            Ok(())
        }
    }

    #[test]
    fn downstream_failure_cancels_future_source_reads() {
        let samples = [1.0, 2.0, 3.0];
        let source = SliceAudioSource::new(&samples);
        let cancelled = Arc::new(AtomicBool::new(false));
        let guarded = CancellableAudioSource {
            source: &source,
            cancelled: Arc::clone(&cancelled),
        };
        let mut window = [0.0; 2];
        guarded.read_exact_samples(0, &mut window).unwrap();
        assert_eq!(window, [1.0, 2.0]);

        cancelled.store(true, Ordering::Release);
        assert_eq!(
            guarded.read_exact_samples(1, &mut window),
            Err(AudioSourceError::Cancelled)
        );
    }

    #[test]
    fn downstream_failure_drains_full_bounded_bridge_and_joins_workers() {
        const CAPACITY: usize = 100;
        let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let source = CountingSource {
            reads: Arc::clone(&reads),
            sample_count: 2_000,
        };
        let sample_count = source.sample_count();
        let cancelled = Arc::new(AtomicBool::new(false));
        let guarded = CancellableAudioSource {
            source: &source,
            cancelled: Arc::clone(&cancelled),
        };
        let (segment_tx, segment_rx) = crossbeam_channel::bounded::<usize>(CAPACITY);
        let (decoded_tx, decoded_rx) = crossbeam_channel::bounded::<usize>(CAPACITY);
        let segment_observer = segment_rx.clone();
        let mut decoded_observer = Some(decoded_rx.clone());

        std::thread::scope(|scope| {
            let producer = scope.spawn(move || {
                for start in 0..sample_count {
                    let mut sample = [0.0];
                    if let Err(error) = guarded.read_exact_samples(start, &mut sample) {
                        return Err(error);
                    }
                    if segment_tx.send(start).is_err() {
                        return Ok(());
                    }
                }
                Ok(())
            });
            let bridge = scope.spawn(move || {
                while let Ok(segment) = segment_rx.recv() {
                    if decoded_tx.send(segment).is_err() {
                        break;
                    }
                }
            });

            let fill_deadline = Instant::now() + Duration::from_secs(2);
            while (segment_observer.len() < CAPACITY
                || decoded_observer.as_ref().unwrap().len() < CAPACITY)
                && Instant::now() < fill_deadline
            {
                std::thread::sleep(Duration::from_millis(1));
            }
            let both_stages_full = segment_observer.len() == CAPACITY
                && decoded_observer.as_ref().unwrap().len() == CAPACITY;
            let read_count_before_cancel = reads.load(Ordering::Relaxed);

            // This models the source pipeline returning an injected downstream read/model failure.
            let consumer_result: Result<(), AudioSourceError> = Err(AudioSourceError::ReadFailed);
            assert_eq!(consumer_result, Err(AudioSourceError::ReadFailed));
            let (drained_tx, drained_rx) = crossbeam_channel::bounded(1);
            let cancel_token = Arc::clone(&cancelled);
            let drain = scope.spawn(move || {
                let drained =
                    cancel_source_and_drain_on_error(&consumer_result, &decoded_rx, &cancel_token);
                let _ = drained_tx.send(drained);
                drained
            });

            let drained_count = drained_rx.recv_timeout(Duration::from_secs(3)).ok();
            if drained_count.is_none_or(|count| count < CAPACITY) {
                // Ensure a regression fails promptly and releases both bounded senders.
                cancelled.store(true, Ordering::Release);
                drop(decoded_observer.take());
            }
            let producer_result = producer.join().expect("source producer joined");
            bridge.join().expect("bridge joined");
            let drain_count = drain.join().expect("bounded-channel drain joined");

            assert!(
                both_stages_full,
                "both capacity-100 queues should fill before failure"
            );
            assert!(read_count_before_cancel > CAPACITY * 2);
            assert!(
                drained_count.is_some_and(|count| count >= CAPACITY),
                "drain must finish and consume the full decoded queue within the direct test deadline"
            );
            assert_eq!(drain_count, drained_count.unwrap_or_default());
            assert_eq!(producer_result, Err(AudioSourceError::Cancelled));
            assert!(cancelled.load(Ordering::Acquire));
        });
    }
}

struct EmbeddingSummary {
    segmentations: Array3<f32>,
    embeddings: Array3<f32>,
    num_chunks: usize,
    gpu_predict_us: u64,
    prep_fbank_us: u64,
    prep_mask_us: u64,
}

fn join_scoped_result<T>(
    name: &str,
    handle: std::thread::ScopedJoinHandle<'_, Result<T, PipelineError>>,
) -> Result<T, PipelineError> {
    handle.join().map_err(|_| worker_panic(name))?
}

fn chunk_session_for_windows(
    emb_model: &mut EmbeddingModel,
    wins: usize,
) -> Result<&ChunkEmbeddingSession, PipelineError> {
    emb_model.chunk_session_for_windows(wins)?.ok_or_else(|| {
        invariant_error(format!(
            "missing chunk embedding session for {wins} windows"
        ))
    })
}

pub(super) fn try_chunk_embedding(
    seg_model: &mut SegmentationModel,
    emb_model: &mut EmbeddingModel,
    powerset: &PowersetMapping,
    audio: &[f32],
) -> Result<Option<InferenceArtifacts>, PipelineError> {
    let Some((chunk_win_capacity, total_windows, use_pipelined, chunk_resources)) =
        setup_chunk_embedding(seg_model, emb_model, audio)?
    else {
        return Ok(None);
    };

    let inference_start = std::time::Instant::now();
    let step_seconds = seg_model.step_seconds();
    let params = ChunkParams {
        step_samples: seg_model.step_samples(),
        window_samples: seg_model.window_samples(),
        num_speakers: 3,
        min_num_samples: emb_model.min_num_samples(),
        chunk_win_capacity,
        total_windows,
    };

    let (seg_tx, seg_rx) = crossbeam_channel::bounded::<Array2<f32>>(100);
    let (chunk_tx, chunk_rx) = crossbeam_channel::bounded::<DecodedChunk>(100);

    std::thread::scope(|scope| {
        let seg_start = std::time::Instant::now();
        let seg_warm_start_windows = chunk_win_capacity;
        let seg_handle = scope.spawn(move || -> Result<std::time::Duration, PipelineError> {
            seg_model.run_streaming_parallel(
                audio,
                seg_tx,
                seg_worker_count(),
                Some(seg_warm_start_windows),
            )?;
            Ok(seg_start.elapsed())
        });

        let bridge_handle = scope.spawn(move || {
            let mut group = Vec::with_capacity(chunk_win_capacity);
            let mut global_start = 0usize;

            for raw_window in &seg_rx {
                group.push(powerset.hard_decode(&raw_window));

                if group.len() == chunk_win_capacity {
                    if chunk_tx
                        .send(DecodedChunk {
                            global_start,
                            decoded_chunk: std::mem::take(&mut group),
                        })
                        .is_err()
                    {
                        break;
                    }
                    global_start += chunk_win_capacity;
                    group = Vec::with_capacity(chunk_win_capacity);
                }
            }

            if !group.is_empty() {
                let _ = chunk_tx.send(DecodedChunk {
                    global_start,
                    decoded_chunk: group,
                });
            }
        });

        let emb_start = std::time::Instant::now();
        let summary = if use_pipelined {
            let Some(resources) = chunk_resources else {
                return Ok(None);
            };
            run_pipelined(scope, emb_start, resources, &chunk_rx, audio, &params)?
        } else {
            run_sequential_chunks(emb_model, &chunk_rx, audio, &params, emb_start)?
        };
        let emb_elapsed = emb_start.elapsed();

        let seg_thread_elapsed = seg_handle
            .join()
            .map_err(|_| worker_panic("segmentation"))??;
        bridge_handle
            .join()
            .map_err(|_| worker_panic("segmentation bridge"))?;
        trace!(
            seg_thread_ms = seg_thread_elapsed.as_millis(),
            seg_wall_ms = seg_start.elapsed().as_millis(),
            "SEG timing"
        );

        let num_chunks = summary.num_chunks;
        let gpu_predict_us = summary.gpu_predict_us;
        let prep_fbank_us = summary.prep_fbank_us;
        let prep_mask_us = summary.prep_mask_us;
        let Some(artifacts) = build_chunk_artifacts(
            step_seconds,
            params.step_samples,
            params.window_samples,
            summary,
        ) else {
            return Ok(None);
        };

        let inference_elapsed = inference_start.elapsed();
        let audio_secs = audio.len() as f64 / 16_000.0;
        debug!(
            chunks = num_chunks,
            chunk_capacity = params.chunk_win_capacity,
            pipelined = use_pipelined,
            seg_ms = seg_thread_elapsed.as_millis(),
            emb_ms = emb_elapsed.as_millis(),
            predict_ms = gpu_predict_us / 1000,
            prep_fbank_ms = prep_fbank_us / 1000,
            prep_mask_ms = prep_mask_us / 1000,
            total_ms = inference_elapsed.as_millis(),
            audio_secs = audio_secs as u64,
            "Chunk embedding complete"
        );

        Ok(Some(artifacts))
    })
}

/// Preserve the CoreML chunk pipeline for file-backed sources. The source workers retain only a
/// chunk-sized waveform scratch; segmentation and embedding keep the same bounded channels,
/// model sessions, chunk capacity and result ordering as the slice path.
pub(super) fn try_chunk_embedding_source<S: AudioSource + ?Sized>(
    seg_model: &mut SegmentationModel,
    emb_model: &mut EmbeddingModel,
    powerset: &PowersetMapping,
    source: &S,
) -> Result<Option<InferenceArtifacts>, PipelineError> {
    let Some((chunk_win_capacity, total_windows, use_pipelined, resources)) =
        setup_chunk_embedding_for_len(seg_model, emb_model, source.sample_count())?
    else {
        return Ok(None);
    };
    if use_pipelined && resources.is_none() {
        return Ok(None);
    }
    let inference_start = std::time::Instant::now();
    let step_seconds = seg_model.step_seconds();
    let params = ChunkParams {
        step_samples: seg_model.step_samples(),
        window_samples: seg_model.window_samples(),
        num_speakers: 3,
        min_num_samples: emb_model.min_num_samples(),
        chunk_win_capacity,
        total_windows,
    };
    let (seg_tx, seg_rx) = crossbeam_channel::bounded::<Array2<f32>>(100);
    let (chunk_tx, chunk_rx) = crossbeam_channel::bounded::<DecodedChunk>(100);
    let cancelled = Arc::new(AtomicBool::new(false));
    let guarded_source = CancellableAudioSource {
        source,
        cancelled: Arc::clone(&cancelled),
    };
    let result = std::thread::scope(|scope| {
        let seg_model = &mut *seg_model;
        let source = &guarded_source;
        let seg_tx_worker = seg_tx;
        let segmentation = scope.spawn(move || {
            seg_model.run_streaming_parallel_source(
                source,
                seg_tx_worker,
                seg_worker_count(),
                Some(chunk_win_capacity),
            )
        });
        let bridge = scope.spawn(move || {
            let mut group = Vec::with_capacity(chunk_win_capacity);
            let mut global_start = 0usize;
            for raw_window in &seg_rx {
                group.push(powerset.hard_decode(&raw_window));
                if group.len() == chunk_win_capacity {
                    if chunk_tx
                        .send(DecodedChunk {
                            global_start,
                            decoded_chunk: std::mem::take(&mut group),
                        })
                        .is_err()
                    {
                        break;
                    }
                    global_start += chunk_win_capacity;
                    group = Vec::with_capacity(chunk_win_capacity);
                }
            }
            if !group.is_empty() {
                let _ = chunk_tx.send(DecodedChunk {
                    global_start,
                    decoded_chunk: group,
                });
            }
        });
        let emb_start = std::time::Instant::now();
        let summary = if use_pipelined {
            let resources =
                resources.expect("pipelined source resources were checked before spawn");
            run_pipelined_source(scope, emb_start, resources, &chunk_rx, source, &params)
        } else {
            run_sequential_chunks_source(emb_model, &chunk_rx, source, &params, emb_start)
        };
        if summary.is_err() {
            // Stop source-window reads and drain the bounded bridge output. Otherwise the bridge
            // can block sending while segmentation blocks sending to the bridge input channel.
            cancel_source_and_drain_on_error(&summary, &chunk_rx, &cancelled);
        }
        let segmentation = segmentation
            .join()
            .map_err(|_| worker_panic("source chunk segmentation"))?;
        let bridge = bridge
            .join()
            .map_err(|_| worker_panic("source segmentation bridge"));
        // Drain/joins above are unconditional so read, backend, cancellation and worker failures
        // cannot leave the segmentation producers detached.
        let summary = summary?;
        segmentation?;
        bridge?;
        let artifacts = build_chunk_artifacts(
            step_seconds,
            params.step_samples,
            params.window_samples,
            summary,
        );
        Ok::<_, PipelineError>(artifacts)
    })?;
    let inference_elapsed = inference_start.elapsed();
    let Some(artifacts) = result else {
        return Ok(None);
    };
    debug!(
        chunks = artifacts.segmentations.shape()[0],
        windows = total_windows,
        audio_secs = source.sample_count() as f64 / 16_000.0,
        inference_ms = inference_elapsed.as_millis(),
        "Source CoreML chunk embedding complete"
    );
    Ok(Some(artifacts))
}

pub(super) fn try_batch_chunk_embedding(
    seg_model: &mut SegmentationModel,
    emb_model: &mut EmbeddingModel,
    powerset: &PowersetMapping,
    plda: &PldaTransform,
    files: &[BatchInput<'_>],
    config: &PipelineConfig,
) -> Result<Option<Vec<DiarizationResult>>, PipelineError> {
    if files.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let chunk_win_capacity = match emb_model.chunk_window_capacity() {
        Some(capacity) => capacity,
        None => return Ok(None),
    };

    let step_samples = seg_model.step_samples();
    let window_samples = seg_model.window_samples();
    let step_seconds = seg_model.step_seconds();
    let num_speakers = 3usize;
    let min_num_samples = emb_model.min_num_samples();

    if files.iter().any(|file| file.audio.len() < window_samples) {
        return Ok(None);
    }

    let expected_chunks: Vec<usize> = files
        .iter()
        .map(|file| {
            let total_windows = file.audio.len().saturating_sub(window_samples) / step_samples + 1;
            total_windows.div_ceil(chunk_win_capacity)
        })
        .collect();

    if expected_chunks.iter().sum::<usize>() < 2 {
        return Ok(None);
    }

    let Some(resources) = chunk_embedding_resources(emb_model)? else {
        return Ok(None);
    };

    let largest = resources.largest_session()?;
    let prep_config = ChunkPrep {
        step_samples,
        window_samples,
        num_speakers,
        min_num_samples,
        largest_fbank_frames: largest.fbank_frames,
        largest_num_masks: largest.num_masks,
        max_active: chunk_win_capacity * num_speakers,
        fbank_30s: resources.fbank_30s.clone(),
        fbank_10s: resources.fbank_10s.clone(),
    };

    let audios: Vec<&[f32]> = files.iter().map(|file| file.audio).collect();
    let batch_start = std::time::Instant::now();

    let (decoded_tx, decoded_rx) = crossbeam_channel::bounded::<TaggedDecoded>(100);
    let (prepared_tx, prepared_rx) = crossbeam_channel::bounded::<TaggedPrepared>(48);
    let (embedded_tx, embedded_rx) = crossbeam_channel::bounded::<TaggedEmbedded>(16);

    std::thread::scope(|scope| {
        let audios_ref = &audios;

        let decoded_tx_seg = decoded_tx.clone();
        let seg_handle = scope.spawn(move || -> Result<(), PipelineError> {
            for (file_idx, file) in files.iter().enumerate() {
                let (seg_tx, seg_rx) = crossbeam_channel::bounded::<Array2<f32>>(100);

                std::thread::scope(|inner| {
                    let decoded_tx_bridge = &decoded_tx_seg;
                    let bridge_handle = inner.spawn(move || {
                        let mut group = Vec::with_capacity(chunk_win_capacity);
                        let mut local_start = 0usize;

                        for raw_window in &seg_rx {
                            group.push(powerset.hard_decode(&raw_window));
                            if group.len() == chunk_win_capacity {
                                if decoded_tx_bridge
                                    .send(TaggedDecoded {
                                        file_idx,
                                        local_start,
                                        decoded_chunk: std::mem::take(&mut group),
                                    })
                                    .is_err()
                                {
                                    return;
                                }
                                local_start += 1;
                                group = Vec::with_capacity(chunk_win_capacity);
                            }
                        }

                        if !group.is_empty() {
                            let _ = decoded_tx_bridge.send(TaggedDecoded {
                                file_idx,
                                local_start,
                                decoded_chunk: group,
                            });
                        }
                    });

                    seg_model.run_streaming_parallel(
                        file.audio,
                        seg_tx,
                        seg_worker_count(),
                        Some(chunk_win_capacity),
                    )?;

                    bridge_handle
                        .join()
                        .map_err(|_| worker_panic("batch segmentation bridge"))?;
                    Ok::<(), PipelineError>(())
                })?;
            }
            drop(decoded_tx_seg);
            Ok(())
        });
        drop(decoded_tx);

        let mut prep_handles = Vec::with_capacity(2);
        for _ in 0..2usize {
            let worker = BatchPrepWorker {
                prep: prep_config.clone(),
                scratch: PrepScratch::new(window_samples),
            };
            let prepared_tx = prepared_tx.clone();
            let decoded_rx = decoded_rx.clone();
            prep_handles
                .push(scope.spawn(move || worker.run(audios_ref, &decoded_rx, prepared_tx)));
        }
        drop(prepared_tx);

        let gpu_worker = BatchGpuWorker {
            model: largest.session.model,
            fbank_shape: largest.session.cached_fbank_shape,
            masks_shape: largest.session.cached_masks_shape,
            prep: prep_config,
            scratch: PrepScratch::new(window_samples),
        };
        let gpu_embedded_tx = embedded_tx.clone();
        let gpu_decoded_rx = decoded_rx.clone();
        let gpu_handle = scope.spawn(move || {
            gpu_worker.run(audios_ref, prepared_rx, gpu_decoded_rx, gpu_embedded_tx)
        });
        drop(embedded_tx);

        let expected_windows: Vec<usize> = files
            .iter()
            .map(|file| file.audio.len().saturating_sub(window_samples) / step_samples + 1)
            .collect();
        let mut collectors: Vec<Option<FileCollector>> =
            std::iter::repeat_with(|| None).take(files.len()).collect();
        let mut results: Vec<Option<DiarizationResult>> =
            std::iter::repeat_with(|| None).take(files.len()).collect();
        let mut files_complete = 0usize;

        for tagged in std::iter::from_fn(|| embedded_rx.recv().ok()) {
            let collector = collectors[tagged.file_idx].get_or_insert_with(|| {
                let num_frames = tagged.embedded.decoded_chunk[0].nrows();
                let max_slots = expected_windows[tagged.file_idx] + chunk_win_capacity;
                FileCollector::new(
                    max_slots,
                    num_frames,
                    num_speakers,
                    expected_chunks[tagged.file_idx],
                )
            });

            collector.add(
                tagged.local_start,
                chunk_win_capacity,
                num_speakers,
                tagged.embedded,
            )?;

            if collector.is_complete() {
                let collector = collectors[tagged.file_idx].take().ok_or_else(|| {
                    invariant_error(format!(
                        "collector for file {} completed without state",
                        tagged.file_idx
                    ))
                })?;
                if let Some(artifacts) =
                    collector.into_artifacts(step_seconds, step_samples, window_samples)
                {
                    results[tagged.file_idx] = Some(post_inference(artifacts, config, plda)?);
                }
                files_complete += 1;
            }
        }

        join_scoped_result("batch segmentation", seg_handle)?;
        let _gpu_stats = join_scoped_result("batch chunk embedding gpu", gpu_handle)?;
        for handle in prep_handles {
            join_scoped_result("batch chunk embedding prep", handle)?;
        }

        debug!(
            files = files.len(),
            files_complete,
            batch_ms = batch_start.elapsed().as_millis(),
            "Batch chunk embedding complete"
        );

        Ok(results
            .into_iter()
            .map(|result| {
                result.unwrap_or_else(|| DiarizationResult {
                    segmentations: DecodedSegmentations(Array3::zeros((0, 0, num_speakers))),
                    embeddings: ChunkEmbeddings(Array3::from_elem(
                        (0, num_speakers, 256),
                        f32::NAN,
                    )),
                    speaker_count: SpeakerCountTrack(Vec::new()),
                    hard_clusters: ChunkSpeakerClusters(Array2::zeros((0, 0))),
                    discrete_diarization: DiscreteDiarization(Array2::zeros((0, 0))),
                    segments: Vec::new(),
                })
            })
            .collect())
    })
    .map(Some)
}
