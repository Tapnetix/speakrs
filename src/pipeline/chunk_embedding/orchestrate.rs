use crate::audio_source::AudioSource;
use crossbeam_channel::Receiver;
use ndarray::{Array3, s};
use tracing::{debug, trace};

use super::collect::batch_embeddings;
use super::gpu::{ChunkEmbeddingResources, GpuWorker, chunk_embedding_resources};
use super::prep::{ChunkPrep, DecodedChunk, PrepScratch, PrepWorker, read_source_chunk};
use super::{
    ChunkParams, EmbeddingModel, EmbeddingSummary, PipelineError, SegmentationModel,
    chunk_audio_raw, chunk_session_for_windows, join_scoped_result, write_speaker_mask_to_slice,
};

type ChunkEmbeddingSetup = (usize, usize, bool, Option<ChunkEmbeddingResources>);

pub(super) fn seg_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4)
        .min(8)
}

pub(super) fn setup_chunk_embedding(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
) -> Result<Option<ChunkEmbeddingSetup>, PipelineError> {
    setup_chunk_embedding_for_len(seg_model, emb_model, audio.len())
}

pub(super) fn setup_chunk_embedding_for_len(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    sample_count: usize,
) -> Result<Option<ChunkEmbeddingSetup>, PipelineError> {
    let window_samples = seg_model.window_samples();
    if sample_count < window_samples {
        return Ok(None);
    }

    let Some(chunk_win_capacity) = emb_model.chunk_window_capacity() else {
        return Ok(None);
    };
    let step_samples = seg_model.step_samples();
    let total_windows = chunk_embedding_window_count(sample_count, window_samples, step_samples);
    if total_windows == 0 {
        return Ok(None);
    }
    let est_chunks = total_windows.div_ceil(chunk_win_capacity);
    let use_pipelined = est_chunks >= 2;
    let resources = use_pipelined
        .then(|| chunk_embedding_resources(emb_model))
        .transpose()?
        .flatten();

    Ok(Some((
        chunk_win_capacity,
        total_windows,
        use_pipelined,
        resources,
    )))
}

fn chunk_embedding_window_count(
    sample_count: usize,
    window_samples: usize,
    step_samples: usize,
) -> usize {
    if window_samples == 0 || step_samples == 0 || sample_count < window_samples {
        return 0;
    }
    sample_count.saturating_sub(window_samples) / step_samples + 1
}

fn source_chunk_has_complete_first_window(
    sample_count: usize,
    global_start: usize,
    step_samples: usize,
    window_samples: usize,
) -> Result<bool, PipelineError> {
    let start = global_start
        .checked_mul(step_samples)
        .ok_or_else(|| super::invariant_error("source chunk offset overflowed"))?;
    let end = start
        .checked_add(window_samples)
        .ok_or_else(|| super::invariant_error("source chunk window end overflowed"))?;
    Ok(end <= sample_count)
}

#[cfg(test)]
mod source_geometry_tests {
    use super::{chunk_embedding_window_count, source_chunk_has_complete_first_window};

    #[test]
    fn source_chunk_geometry_keeps_full_window_and_batch_counts() {
        let window = 160_000;
        let step = 80_000;
        for (samples, expected_windows, expected_32_batches, expected_64_batches) in [
            (0, 0, 0, 0),
            (window - 1, 0, 0, 0),
            (window, 1, 1, 1),
            (window + step, 2, 1, 1),
            (window + step * 63, 64, 2, 1),
            (window + step * 64 + 1, 65, 3, 2),
        ] {
            let windows = chunk_embedding_window_count(samples, window, step);
            assert_eq!(windows, expected_windows, "sample_count={samples}");
            assert_eq!(windows.div_ceil(32), expected_32_batches);
            assert_eq!(windows.div_ceil(64), expected_64_batches);
        }
    }

    #[test]
    fn source_sequential_fallback_skips_tail_chunk_like_slice_path() {
        let window = 160_000;
        let step = 80_000;
        let capacity = 116;
        // This produces exactly `capacity` complete windows plus a padded final window
        // that starts a new chunk. The slice sequential path skips that incomplete chunk.
        let sample_count = window + (capacity - 1) * step + 1;
        assert_eq!(
            chunk_embedding_window_count(sample_count, window, step),
            capacity
        );
        assert!(source_chunk_has_complete_first_window(sample_count, 0, step, window).unwrap());
        assert!(
            source_chunk_has_complete_first_window(sample_count, capacity - 1, step, window)
                .unwrap()
        );
        assert!(
            !source_chunk_has_complete_first_window(sample_count, capacity, step, window).unwrap()
        );
    }
}

pub(super) fn run_pipelined<'scope>(
    scope: &'scope std::thread::Scope<'scope, '_>,
    emb_start: std::time::Instant,
    resources: ChunkEmbeddingResources,
    chunk_rx: &'scope Receiver<DecodedChunk>,
    audio: &'scope [f32],
    params: &'scope ChunkParams,
) -> Result<EmbeddingSummary, PipelineError> {
    let largest = resources.largest_session()?;
    let max_active = params.chunk_win_capacity * params.num_speakers;
    let prep_config = ChunkPrep {
        step_samples: params.step_samples,
        window_samples: params.window_samples,
        num_speakers: params.num_speakers,
        min_num_samples: params.min_num_samples,
        largest_fbank_frames: largest.fbank_frames,
        largest_num_masks: largest.num_masks,
        max_active,
        fbank_30s: resources.fbank_30s.clone(),
        fbank_10s: resources.fbank_10s.clone(),
    };

    let (prep_tx, prep_rx) = crossbeam_channel::bounded(48);
    let (emb_tx, emb_rx) = crossbeam_channel::bounded(8);

    let mut prep_handles = Vec::with_capacity(2);
    for _ in 0..2usize {
        let worker = PrepWorker {
            prep: prep_config.clone(),
            scratch: PrepScratch::new(params.window_samples),
        };
        let prep_tx = prep_tx.clone();
        prep_handles.push(scope.spawn(move || {
            worker.run(
                audio,
                params.step_samples,
                params.window_samples,
                chunk_rx,
                prep_tx,
            )
        }));
    }
    drop(prep_tx);

    let gpu_emb_tx = emb_tx.clone();
    let gpu_handle = scope.spawn(move || {
        GpuWorker {
            model: largest.session.model,
            fbank_shape: largest.session.cached_fbank_shape,
            masks_shape: largest.session.cached_masks_shape,
            prep: prep_config,
            scratch: PrepScratch::new(params.window_samples),
        }
        .run(audio, prep_rx, chunk_rx.clone(), gpu_emb_tx)
    });
    drop(emb_tx);

    let max_slots = params.total_windows + params.chunk_win_capacity;
    let first = match emb_rx.recv() {
        Ok(embedded) => embedded,
        Err(_) => {
            let _gpu_stats = join_scoped_result("chunk embedding gpu", gpu_handle)?;
            let mut total_prep_fbank_us = 0u64;
            for handle in prep_handles {
                total_prep_fbank_us += join_scoped_result("chunk embedding prep", handle)?.fbank_us;
            }
            return Ok(EmbeddingSummary {
                segmentations: Array3::zeros((0, 0, params.num_speakers)),
                embeddings: Array3::from_elem((0, params.num_speakers, 256), f32::NAN),
                num_chunks: 0,
                gpu_predict_us: 0,
                prep_fbank_us: total_prep_fbank_us,
                prep_mask_us: 0,
            });
        }
    };

    let num_frames = first.decoded_chunk[0].nrows();
    let mut seg_array = Array3::<f32>::zeros((max_slots, num_frames, params.num_speakers));
    let mut emb_array = Array3::<f32>::from_elem((max_slots, params.num_speakers, 256), f32::NAN);
    let mut max_slot_used = 0usize;
    let mut total_predict_us = 0u64;
    let mut total_chunks = 0u32;

    for embedded in std::iter::once(first).chain(std::iter::from_fn(|| emb_rx.recv().ok())) {
        total_predict_us += embedded.predict_us;
        let batch_emb = batch_embeddings(
            embedded.num_masks,
            embedded.data,
            "pipelined chunk embedding",
        )?;

        for &(local, speaker_idx) in &embedded.active {
            let slot = embedded.global_start + local;
            if slot < max_slots {
                let mask_idx = local * params.num_speakers + speaker_idx;
                emb_array
                    .slice_mut(s![slot, speaker_idx, ..])
                    .assign(&batch_emb.row(mask_idx));
            }
        }

        for (local, decoded) in embedded.decoded_chunk.into_iter().enumerate() {
            let slot = embedded.global_start + local;
            if slot < max_slots {
                seg_array.slice_mut(s![slot, .., ..]).assign(&decoded);
                max_slot_used = max_slot_used.max(slot + 1);
            }
        }

        total_chunks += 1;
    }

    let gpu_stats = join_scoped_result("chunk embedding gpu", gpu_handle)?;
    let mut total_prep_fbank_us = 0u64;
    for handle in prep_handles {
        total_prep_fbank_us += join_scoped_result("chunk embedding prep", handle)?.fbank_us;
    }

    let num_chunks = max_slot_used;
    let seg_array = seg_array.slice_move(s![..num_chunks, .., ..]);
    let emb_array = emb_array.slice_move(s![..num_chunks, .., ..]);

    debug!(
        total_chunks,
        num_chunks,
        gpu_chunks = gpu_stats.chunks,
        gpu_predict_ms = gpu_stats.predict_us / 1000,
        gpu_self_prep_ms = gpu_stats.self_prep_us / 1000,
        cpu_prep_ms = total_prep_fbank_us / 1000,
        predict_ms = total_predict_us / 1000,
        emb_wall_ms = emb_start.elapsed().as_millis(),
        "Chunk embedding worker breakdown"
    );

    Ok(EmbeddingSummary {
        segmentations: seg_array,
        embeddings: emb_array,
        num_chunks,
        gpu_predict_us: total_predict_us,
        prep_fbank_us: total_prep_fbank_us + gpu_stats.self_prep_us,
        prep_mask_us: 0,
    })
}

pub(super) fn run_pipelined_source<'scope, S: AudioSource + ?Sized>(
    scope: &'scope std::thread::Scope<'scope, '_>,
    emb_start: std::time::Instant,
    resources: ChunkEmbeddingResources,
    chunk_rx: &'scope Receiver<DecodedChunk>,
    source: &'scope S,
    params: &'scope ChunkParams,
) -> Result<EmbeddingSummary, PipelineError> {
    let largest = resources.largest_session()?;
    let prep_config = ChunkPrep {
        step_samples: params.step_samples,
        window_samples: params.window_samples,
        num_speakers: params.num_speakers,
        min_num_samples: params.min_num_samples,
        largest_fbank_frames: largest.fbank_frames,
        largest_num_masks: largest.num_masks,
        max_active: params.chunk_win_capacity * params.num_speakers,
        fbank_30s: resources.fbank_30s.clone(),
        fbank_10s: resources.fbank_10s.clone(),
    };
    let (prep_tx, prep_rx) = crossbeam_channel::bounded(48);
    let (emb_tx, emb_rx) = crossbeam_channel::bounded(8);
    let mut prep_handles = Vec::with_capacity(2);
    for _ in 0..2 {
        let worker = PrepWorker {
            prep: prep_config.clone(),
            scratch: PrepScratch::new(params.window_samples),
        };
        let prep_tx = prep_tx.clone();
        prep_handles.push(scope.spawn(move || worker.run_source(source, chunk_rx, prep_tx)));
    }
    drop(prep_tx);
    let gpu_tx = emb_tx.clone();
    let gpu_handle = scope.spawn(move || {
        GpuWorker {
            model: largest.session.model,
            fbank_shape: largest.session.cached_fbank_shape,
            masks_shape: largest.session.cached_masks_shape,
            prep: prep_config,
            scratch: PrepScratch::new(params.window_samples),
        }
        .run_source(source, prep_rx, chunk_rx.clone(), gpu_tx)
    });
    drop(emb_tx);

    let max_slots = params.total_windows + params.chunk_win_capacity;
    let first = match emb_rx.recv() {
        Ok(embedded) => embedded,
        Err(_) => {
            let _ = join_scoped_result("source chunk embedding gpu", gpu_handle)?;
            for handle in prep_handles {
                let _ = join_scoped_result("source chunk embedding prep", handle)?;
            }
            return Ok(EmbeddingSummary {
                segmentations: Array3::zeros((0, 0, params.num_speakers)),
                embeddings: Array3::from_elem((0, params.num_speakers, 256), f32::NAN),
                num_chunks: 0,
                gpu_predict_us: 0,
                prep_fbank_us: 0,
                prep_mask_us: 0,
            });
        }
    };
    let frames = first.decoded_chunk[0].nrows();
    let mut seg_array = Array3::<f32>::zeros((max_slots, frames, params.num_speakers));
    let mut emb_array = Array3::<f32>::from_elem((max_slots, params.num_speakers, 256), f32::NAN);
    let mut max_slot_used = 0usize;
    let mut predict_us = 0u64;
    let mut total_chunks = 0u32;
    for embedded in std::iter::once(first).chain(std::iter::from_fn(|| emb_rx.recv().ok())) {
        predict_us += embedded.predict_us;
        let batch_emb =
            batch_embeddings(embedded.num_masks, embedded.data, "source chunk embedding")?;
        for &(local, speaker) in &embedded.active {
            let slot = embedded.global_start + local;
            if slot < max_slots {
                let mask = local * params.num_speakers + speaker;
                emb_array
                    .slice_mut(s![slot, speaker, ..])
                    .assign(&batch_emb.row(mask));
            }
        }
        for (local, decoded) in embedded.decoded_chunk.into_iter().enumerate() {
            let slot = embedded.global_start + local;
            if slot < max_slots {
                seg_array.slice_mut(s![slot, .., ..]).assign(&decoded);
                max_slot_used = max_slot_used.max(slot + 1);
            }
        }
        total_chunks += 1;
    }
    let gpu_stats = join_scoped_result("source chunk embedding gpu", gpu_handle)?;
    let mut prep_us = 0u64;
    for handle in prep_handles {
        prep_us += join_scoped_result("source chunk embedding prep", handle)?.fbank_us;
    }
    let num_chunks = max_slot_used;
    debug!(
        total_chunks,
        num_chunks,
        gpu_chunks = gpu_stats.chunks,
        gpu_predict_ms = gpu_stats.predict_us / 1000,
        cpu_prep_ms = prep_us / 1000,
        predict_ms = predict_us / 1000,
        emb_wall_ms = emb_start.elapsed().as_millis(),
        "Source chunk embedding worker breakdown"
    );
    Ok(EmbeddingSummary {
        segmentations: seg_array.slice_move(s![..num_chunks, .., ..]),
        embeddings: emb_array.slice_move(s![..num_chunks, .., ..]),
        num_chunks,
        gpu_predict_us: predict_us,
        prep_fbank_us: prep_us + gpu_stats.self_prep_us,
        prep_mask_us: 0,
    })
}

pub(super) fn run_sequential_chunks(
    emb_model: &mut EmbeddingModel,
    chunk_rx: &Receiver<DecodedChunk>,
    audio: &[f32],
    params: &ChunkParams,
    emb_start: std::time::Instant,
) -> Result<EmbeddingSummary, PipelineError> {
    let max_slots = params.total_windows + params.chunk_win_capacity;
    let mut seg_array: Option<Array3<f32>> = None;
    let mut emb_array: Option<Array3<f32>> = None;
    let mut seq_idx = 0usize;
    let mut seq_fbank_us = 0u64;
    let mut seq_mask_us = 0u64;
    let mut seq_predict_us = 0u64;
    let mut seq_chunks = 0u32;

    for DecodedChunk {
        global_start,
        decoded_chunk,
    } in chunk_rx
    {
        let wins = decoded_chunk.len();
        let session = chunk_session_for_windows(emb_model, wins)?;
        let sess_fbank_frames = session.fbank_frames;
        let sess_num_masks = session.num_masks;

        let chunk_audio_start = global_start * params.step_samples;
        if chunk_audio_start + params.window_samples > audio.len() {
            continue;
        }
        let chunk_audio_len = params.window_samples + (wins - 1) * params.step_samples;
        let chunk_audio_end = (chunk_audio_start + chunk_audio_len).min(audio.len());
        let chunk_audio = &audio[chunk_audio_start..chunk_audio_end];

        let num_frames = decoded_chunk[0].nrows();
        let seg = seg_array
            .get_or_insert_with(|| Array3::zeros((max_slots, num_frames, params.num_speakers)));
        let emb = emb_array.get_or_insert_with(|| {
            Array3::from_elem((max_slots, params.num_speakers, 256), f32::NAN)
        });

        let mut fbank = vec![0.0f32; sess_fbank_frames * 80];
        let fbank_start = std::time::Instant::now();

        if let Some(full_fbank) = emb_model.compute_chunk_fbank_30s(chunk_audio)? {
            let copy_frames = full_fbank.nrows().min(sess_fbank_frames);
            for row_idx in 0..copy_frames {
                let dst = row_idx * 80;
                let row_view = full_fbank.row(row_idx);
                let row = row_view.as_slice().ok_or_else(|| {
                    super::invariant_error("30s chunk fbank row was not contiguous")
                })?;
                fbank[dst..dst + 80].copy_from_slice(row);
            }
        } else {
            let mut fbank_offset = 0usize;
            let mut audio_offset = 0usize;
            while fbank_offset < sess_fbank_frames && audio_offset < chunk_audio.len() {
                let segment_end = (audio_offset + params.window_samples).min(chunk_audio.len());
                let seg_fbank =
                    emb_model.compute_chunk_fbank(&chunk_audio[audio_offset..segment_end])?;
                let copy = seg_fbank.nrows().min(sess_fbank_frames - fbank_offset);
                for row_idx in 0..copy {
                    let dst = (fbank_offset + row_idx) * 80;
                    let row_view = seg_fbank.row(row_idx);
                    let row = row_view.as_slice().ok_or_else(|| {
                        super::invariant_error("10s chunk fbank row was not contiguous")
                    })?;
                    fbank[dst..dst + 80].copy_from_slice(row);
                }
                fbank_offset += 998;
                audio_offset += params.window_samples;
            }
        }

        seq_fbank_us += fbank_start.elapsed().as_micros() as u64;
        let mask_start = std::time::Instant::now();
        let mut masks = vec![0.0f32; sess_num_masks * 589];
        let mut active = Vec::new();
        for (local_idx, decoded) in decoded_chunk.iter().enumerate() {
            let global_idx = global_start + local_idx;
            let win_audio = chunk_audio_raw(
                audio,
                params.step_samples,
                params.window_samples,
                global_idx,
            );
            for speaker_idx in 0..params.num_speakers {
                let mask_idx = local_idx * params.num_speakers + speaker_idx;
                if mask_idx >= sess_num_masks {
                    break;
                }
                let dest = &mut masks[mask_idx * 589..mask_idx * 589 + 589];
                if write_speaker_mask_to_slice(
                    &decoded.view(),
                    speaker_idx,
                    win_audio.len(),
                    params.min_num_samples,
                    dest,
                ) {
                    active.push((local_idx, speaker_idx));
                }
            }
        }
        seq_mask_us += mask_start.elapsed().as_micros() as u64;

        let predict_start = std::time::Instant::now();
        let session = chunk_session_for_windows(emb_model, wins)?;
        let batch_emb = EmbeddingModel::embed_chunk_session(session, &fbank, &masks)?;
        seq_predict_us += predict_start.elapsed().as_micros() as u64;
        seq_chunks += 1;

        for &(local_idx, speaker_idx) in &active {
            let mask_idx = local_idx * params.num_speakers + speaker_idx;
            emb.slice_mut(s![seq_idx + local_idx, speaker_idx, ..])
                .assign(&batch_emb.row(mask_idx));
        }
        for (local_idx, decoded) in decoded_chunk.into_iter().enumerate() {
            seg.slice_mut(s![seq_idx + local_idx, .., ..])
                .assign(&decoded);
        }
        seq_idx += wins;
    }

    let num_chunks = seq_idx;
    let seg_array = match seg_array {
        Some(array) => array.slice_move(s![..num_chunks, .., ..]),
        None => Array3::zeros((0, 0, params.num_speakers)),
    };
    let emb_array = match emb_array {
        Some(array) => array.slice_move(s![..num_chunks, .., ..]),
        None => Array3::from_elem((0, params.num_speakers, 256), f32::NAN),
    };

    trace!(
        seq_chunks,
        num_chunks,
        fbank_ms = seq_fbank_us / 1000,
        mask_ms = seq_mask_us / 1000,
        predict_ms = seq_predict_us / 1000,
        wall_ms = emb_start.elapsed().as_millis(),
        "EMB sequential",
    );

    Ok(EmbeddingSummary {
        segmentations: seg_array,
        embeddings: emb_array,
        num_chunks,
        gpu_predict_us: seq_predict_us,
        prep_fbank_us: seq_fbank_us,
        prep_mask_us: seq_mask_us,
    })
}

pub(super) fn run_sequential_chunks_source<S: AudioSource + ?Sized>(
    emb_model: &mut EmbeddingModel,
    chunk_rx: &Receiver<DecodedChunk>,
    source: &S,
    params: &ChunkParams,
    emb_start: std::time::Instant,
) -> Result<EmbeddingSummary, PipelineError> {
    let max_slots = params.total_windows + params.chunk_win_capacity;
    let mut seg_array: Option<Array3<f32>> = None;
    let mut emb_array: Option<Array3<f32>> = None;
    let mut seq_idx = 0usize;
    let mut seq_fbank_us = 0u64;
    let mut seq_mask_us = 0u64;
    let mut seq_predict_us = 0u64;
    let mut seq_chunks = 0u32;

    for DecodedChunk {
        global_start,
        decoded_chunk,
    } in chunk_rx
    {
        if !source_chunk_has_complete_first_window(
            source.sample_count(),
            global_start,
            params.step_samples,
            params.window_samples,
        )? {
            continue;
        }
        let wins = decoded_chunk.len();
        let session = chunk_session_for_windows(emb_model, wins)?;
        let sess_fbank_frames = session.fbank_frames;
        let sess_num_masks = session.num_masks;
        let chunk_audio = read_source_chunk(
            source,
            global_start,
            wins,
            params.step_samples,
            params.window_samples,
        )?;

        let num_frames = decoded_chunk[0].nrows();
        let seg = seg_array
            .get_or_insert_with(|| Array3::zeros((max_slots, num_frames, params.num_speakers)));
        let emb = emb_array.get_or_insert_with(|| {
            Array3::from_elem((max_slots, params.num_speakers, 256), f32::NAN)
        });

        let mut fbank = vec![0.0f32; sess_fbank_frames * 80];
        let fbank_start = std::time::Instant::now();
        if let Some(full_fbank) = emb_model.compute_chunk_fbank_30s(&chunk_audio)? {
            let copy_frames = full_fbank.nrows().min(sess_fbank_frames);
            for row_idx in 0..copy_frames {
                let dst = row_idx * 80;
                let row_view = full_fbank.row(row_idx);
                let row = row_view.as_slice().ok_or_else(|| {
                    super::invariant_error("30s source chunk fbank row was not contiguous")
                })?;
                fbank[dst..dst + 80].copy_from_slice(row);
            }
        } else {
            let mut fbank_offset = 0usize;
            let mut audio_offset = 0usize;
            while fbank_offset < sess_fbank_frames && audio_offset < chunk_audio.len() {
                let segment_end = (audio_offset + params.window_samples).min(chunk_audio.len());
                let seg_fbank =
                    emb_model.compute_chunk_fbank(&chunk_audio[audio_offset..segment_end])?;
                let copy = seg_fbank.nrows().min(sess_fbank_frames - fbank_offset);
                for row_idx in 0..copy {
                    let dst = (fbank_offset + row_idx) * 80;
                    let row_view = seg_fbank.row(row_idx);
                    let row = row_view.as_slice().ok_or_else(|| {
                        super::invariant_error("10s source chunk fbank row was not contiguous")
                    })?;
                    fbank[dst..dst + 80].copy_from_slice(row);
                }
                fbank_offset += 998;
                audio_offset += params.window_samples;
            }
        }
        seq_fbank_us += fbank_start.elapsed().as_micros() as u64;

        let mask_start = std::time::Instant::now();
        let mut masks = vec![0.0f32; sess_num_masks * 589];
        let mut active = Vec::new();
        for (local_idx, decoded) in decoded_chunk.iter().enumerate() {
            let win_audio = chunk_audio_raw(
                &chunk_audio,
                params.step_samples,
                params.window_samples,
                local_idx,
            );
            for speaker_idx in 0..params.num_speakers {
                let mask_idx = local_idx * params.num_speakers + speaker_idx;
                if mask_idx >= sess_num_masks {
                    break;
                }
                let dest = &mut masks[mask_idx * 589..mask_idx * 589 + 589];
                if write_speaker_mask_to_slice(
                    &decoded.view(),
                    speaker_idx,
                    win_audio.len(),
                    params.min_num_samples,
                    dest,
                ) {
                    active.push((local_idx, speaker_idx));
                }
            }
        }
        seq_mask_us += mask_start.elapsed().as_micros() as u64;

        let predict_start = std::time::Instant::now();
        let session = chunk_session_for_windows(emb_model, wins)?;
        let batch_emb = EmbeddingModel::embed_chunk_session(session, &fbank, &masks)?;
        seq_predict_us += predict_start.elapsed().as_micros() as u64;
        seq_chunks += 1;

        for &(local_idx, speaker_idx) in &active {
            let mask_idx = local_idx * params.num_speakers + speaker_idx;
            emb.slice_mut(s![seq_idx + local_idx, speaker_idx, ..])
                .assign(&batch_emb.row(mask_idx));
        }
        for (local_idx, decoded) in decoded_chunk.into_iter().enumerate() {
            seg.slice_mut(s![seq_idx + local_idx, .., ..])
                .assign(&decoded);
        }
        seq_idx += wins;
    }

    let num_chunks = seq_idx;
    let seg_array = match seg_array {
        Some(array) => array.slice_move(s![..num_chunks, .., ..]),
        None => Array3::zeros((0, 0, params.num_speakers)),
    };
    let emb_array = match emb_array {
        Some(array) => array.slice_move(s![..num_chunks, .., ..]),
        None => Array3::from_elem((0, params.num_speakers, 256), f32::NAN),
    };
    trace!(
        seq_chunks,
        num_chunks,
        fbank_ms = seq_fbank_us / 1000,
        mask_ms = seq_mask_us / 1000,
        predict_ms = seq_predict_us / 1000,
        wall_ms = emb_start.elapsed().as_millis(),
        "Source EMB sequential",
    );
    Ok(EmbeddingSummary {
        segmentations: seg_array,
        embeddings: emb_array,
        num_chunks,
        gpu_predict_us: seq_predict_us,
        prep_fbank_us: seq_fbank_us,
        prep_mask_us: seq_mask_us,
    })
}
