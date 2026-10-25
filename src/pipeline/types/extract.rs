use ndarray::{Array2, Array3, s};
use std::sync::Arc;
use tracing::debug;

use crate::audio_source::{AudioSource, AudioSourceError};
use crate::inference::embedding::{EmbeddingModel, MaskedEmbeddingInput, SplitTailInput};
use crate::pipeline::{MIN_SPEAKER_ACTIVITY, clean_masks, select_speaker_weights};
use crate::reconstruct::Reconstructor;

use super::{
    ChunkEmbeddings, ChunkLayout, DecodedSegmentations, EmbeddingPath, PendingEmbedding,
    PendingEmbeddingAudio, PendingSourceEmbedding, PendingSplitEmbedding, PipelineError,
    SpeakerCountTrack,
};

impl DecodedSegmentations {
    pub(in crate::pipeline) fn nchunks(&self) -> usize {
        self.0.shape()[0]
    }

    pub(in crate::pipeline) fn num_speakers(&self) -> usize {
        if self.0.ndim() < 3 {
            return 0;
        }
        self.0.shape()[2]
    }

    pub(in crate::pipeline) fn speaker_count(&self, layout: &ChunkLayout) -> SpeakerCountTrack {
        let reconstructor = Reconstructor::new(self, &layout.start_frames, 0);
        reconstructor.speaker_count(layout.output_frames)
    }

    pub(in crate::pipeline) fn extract_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embedding_path: EmbeddingPath,
    ) -> Result<ChunkEmbeddings, PipelineError> {
        let num_chunks = self.0.shape()[0];
        let num_speakers = self.0.shape()[2];
        let mut embeddings = Array3::<f32>::from_elem((num_chunks, num_speakers, 256), f32::NAN);

        match embedding_path {
            EmbeddingPath::MultiMask => {
                self.extract_multi_mask_embeddings(audio, emb_model, layout, &mut embeddings)?
            }
            EmbeddingPath::Split => {
                self.extract_split_embeddings(audio, emb_model, layout, &mut embeddings)?
            }
            EmbeddingPath::Masked => {
                if emb_model.prefers_chunk_embedding_path() {
                    self.extract_chunk_embeddings(audio, emb_model, layout, &mut embeddings)?;
                } else {
                    self.extract_masked_embeddings(audio, emb_model, layout, &mut embeddings)?;
                }
            }
        }

        Ok(ChunkEmbeddings(embeddings))
    }

    /// Extract embeddings from a random-access source while retaining the model's normal
    /// primary batch schedules. Waveform and fbank scratch is limited to the current batch.
    pub(in crate::pipeline) fn extract_embeddings_source<S: AudioSource + ?Sized>(
        &self,
        source: &S,
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embedding_path: EmbeddingPath,
    ) -> Result<ChunkEmbeddings, PipelineError> {
        let num_chunks = self.0.shape()[0];
        let num_speakers = self.0.shape()[2];
        let mut embeddings = Array3::<f32>::from_elem((num_chunks, num_speakers, 256), f32::NAN);

        match embedding_path {
            EmbeddingPath::MultiMask => {
                self.extract_multi_mask_embeddings_source(
                    source,
                    emb_model,
                    layout,
                    &mut embeddings,
                )?;
            }
            EmbeddingPath::Split => {
                self.extract_split_embeddings_source(source, emb_model, layout, &mut embeddings)?;
            }
            EmbeddingPath::Masked => {
                if emb_model.prefers_chunk_embedding_path() {
                    self.extract_chunk_embeddings_source(
                        source,
                        emb_model,
                        layout,
                        &mut embeddings,
                    )?;
                } else {
                    self.extract_masked_embeddings_source(
                        source,
                        emb_model,
                        layout,
                        &mut embeddings,
                    )?;
                }
            }
        }

        Ok(ChunkEmbeddings(embeddings))
    }

    fn extract_chunk_embeddings_source<S: AudioSource + ?Sized>(
        &self,
        source: &S,
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = read_source_chunk(source, layout, chunk_idx)?;
            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);
            let chunk_embeddings =
                emb_model.embed_chunk_speakers(&chunk_audio, chunk_segmentations, &clean_masks)?;
            embeddings
                .slice_mut(s![chunk_idx, .., ..])
                .assign(&chunk_embeddings);
        }
        Ok(())
    }

    fn extract_masked_embeddings_source<S: AudioSource + ?Sized>(
        &self,
        source: &S,
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.primary_batch_size().max(1);
        let mut storage = Array3Writer(embeddings);
        for active_batch in ActiveSpeakerBatchIter::new(&self.0, batch_size) {
            let mut pending = Vec::with_capacity(active_batch.len());
            let mut cached_chunk = None;
            let mut cached_audio: Option<Arc<[f32]>> = None;
            let mut cached_clean_masks = None;

            for (chunk_idx, speaker_idx) in active_batch {
                if cached_chunk != Some(chunk_idx) {
                    let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
                    cached_audio = Some(read_source_chunk(source, layout, chunk_idx)?.into());
                    cached_clean_masks = Some(clean_masks(&chunk_segmentations));
                    cached_chunk = Some(chunk_idx);
                }
                let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
                let mask = chunk_segmentations.column(speaker_idx);
                let audio = cached_audio.as_ref().ok_or_else(|| {
                    PipelineError::Invariant("source masked batch lost its audio chunk".into())
                })?;
                let clean_masks = cached_clean_masks.as_ref().ok_or_else(|| {
                    PipelineError::Invariant("source masked batch lost its clean masks".into())
                })?;
                pending.push(PendingSourceEmbedding {
                    chunk_idx,
                    speaker_idx,
                    audio: Arc::clone(audio),
                    mask: mask.to_vec(),
                    clean_mask: clean_masks.column(speaker_idx).to_vec(),
                });
            }

            if pending.len() == batch_size {
                flush_source_masked(emb_model, &pending, &mut storage)?;
            } else {
                // Keep the model's historical unbatched tail schedule: a partial 64-row tail
                // is issued as single-row calls when no matching primary session is available.
                while !pending.is_empty() {
                    let batch_len = emb_model.best_batch_len(pending.len()).max(1);
                    flush_source_masked(emb_model, &pending[..batch_len], &mut storage)?;
                    pending.drain(..batch_len);
                }
            }
        }
        Ok(())
    }

    fn extract_split_embeddings_source<S: AudioSource + ?Sized>(
        &self,
        source: &S,
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.split_primary_batch_size();
        if batch_size == 0 {
            return Err(PipelineError::Invariant(
                "source split embeddings require a nonzero primary batch".into(),
            ));
        }
        let num_speakers = self.0.shape()[2];
        let min_num_samples = emb_model.min_num_samples();
        let mut storage = Array3Writer(embeddings);
        let mut pending: Vec<PendingSplitEmbedding> = Vec::with_capacity(batch_size);
        let mut fbanks: Vec<Array2<f32>> = Vec::with_capacity(batch_size);
        let mut tail_batches = 0usize;
        let mut active_items = 0usize;

        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = read_source_chunk(source, layout, chunk_idx)?;
            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);
            let mut active = Vec::with_capacity(num_speakers);

            for speaker_idx in 0..num_speakers {
                let Some(weights) = select_speaker_weights(
                    &chunk_segmentations,
                    &clean_masks,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    continue;
                };
                active.push((speaker_idx, weights));
            }
            if active.is_empty() {
                continue;
            }

            let current_fbank = emb_model.compute_chunk_fbank(&chunk_audio)?;
            fbanks.push(current_fbank.clone());
            let mut current_fbank_idx = fbanks.len() - 1;
            let active_len = active.len();
            for (active_idx, (speaker_idx, weights)) in active.into_iter().enumerate() {
                active_items += 1;
                let full_batch = push_batch_item(
                    &mut pending,
                    PendingSplitEmbedding {
                        chunk_idx,
                        speaker_idx,
                        fbank_idx: current_fbank_idx,
                        weights,
                    },
                    batch_size,
                );
                if full_batch {
                    flush_split(emb_model, &pending, &fbanks, &mut storage)?;
                    tail_batches += 1;
                    pending.clear();
                    fbanks.clear();
                    if active_idx + 1 < active_len {
                        fbanks.push(current_fbank.clone());
                        current_fbank_idx = 0;
                    }
                }
            }
            if pending.is_empty() {
                fbanks.clear();
            }
        }

        if !pending.is_empty() {
            flush_split(emb_model, &pending, &fbanks, &mut storage)?;
            tail_batches += 1;
        }
        debug!(
            batches = tail_batches,
            active_items,
            total_items = self.0.shape()[0] * num_speakers,
            "Split source embeddings complete (batched fbank+tail)"
        );
        Ok(())
    }

    fn extract_multi_mask_embeddings_source<S: AudioSource + ?Sized>(
        &self,
        source: &S,
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.multi_mask_batch_size();
        if batch_size == 0 {
            return Err(PipelineError::Invariant(
                "source multi-mask embeddings require a nonzero batch".into(),
            ));
        }
        let num_speakers = self.0.shape()[2];
        let num_chunks = self.0.shape()[0];
        let min_num_samples = emb_model.min_num_samples();
        let mut storage = Array3Writer(embeddings);
        let mut fbank_buffer = Vec::with_capacity(batch_size);
        let mut masks_buffer = Vec::with_capacity(batch_size * num_speakers);
        let mut chunk_indices = Vec::with_capacity(batch_size);
        let mut batches = 0usize;

        for batch_range in BatchRangeIter::new(num_chunks, batch_size) {
            for chunk_idx in batch_range {
                let chunk_audio = read_source_chunk(source, layout, chunk_idx)?;
                fbank_buffer.push(emb_model.compute_chunk_fbank(&chunk_audio)?);
                chunk_indices.push(chunk_idx);
                let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
                let clean_masks = clean_masks(&chunk_segmentations);
                for speaker_idx in 0..num_speakers {
                    if let Some(weights) = select_speaker_weights(
                        &chunk_segmentations,
                        &clean_masks,
                        speaker_idx,
                        chunk_audio.len(),
                        min_num_samples,
                    ) {
                        masks_buffer.push(weights);
                    } else {
                        masks_buffer.push(vec![0.0; 589]);
                    }
                }
            }

            if !fbank_buffer.is_empty() {
                flush_multi_mask(
                    emb_model,
                    &fbank_buffer,
                    &masks_buffer,
                    &chunk_indices,
                    num_speakers,
                    &mut storage,
                )?;
                batches += 1;
                fbank_buffer.clear();
                masks_buffer.clear();
                chunk_indices.clear();
            }
        }
        debug!(
            batches,
            total_chunks = num_chunks,
            "Multi-mask source embeddings complete"
        );
        Ok(())
    }

    fn extract_chunk_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);
            let chunk_embeddings =
                emb_model.embed_chunk_speakers(chunk_audio, chunk_segmentations, &clean_masks)?;
            embeddings
                .slice_mut(s![chunk_idx, .., ..])
                .assign(&chunk_embeddings);
        }

        Ok(())
    }

    fn extract_masked_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let mut storage = Array3Writer(embeddings);
        let mut pending = Vec::with_capacity(emb_model.primary_batch_size());

        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);

            for speaker_idx in 0..self.0.shape()[2] {
                let mask = chunk_segmentations.column(speaker_idx);
                let activity: f32 = mask.iter().sum();
                if activity < MIN_SPEAKER_ACTIVITY {
                    continue;
                }

                pending.push(PendingEmbedding {
                    chunk_idx,
                    speaker_idx,
                    audio: PendingEmbeddingAudio::Borrowed(chunk_audio),
                    mask: mask.to_vec(),
                    clean_mask: clean_masks.column(speaker_idx).to_vec(),
                });
                if pending.len() == emb_model.primary_batch_size() {
                    flush_masked(emb_model, &pending, &mut storage)?;
                    pending.clear();
                }
            }
        }

        while !pending.is_empty() {
            let batch_len = emb_model.best_batch_len(pending.len());
            flush_masked(emb_model, &pending[..batch_len], &mut storage)?;
            pending.drain(..batch_len);
        }

        Ok(())
    }

    fn extract_split_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.split_primary_batch_size();
        let num_speakers = self.0.shape()[2];
        let min_num_samples = emb_model.min_num_samples();

        let mut storage = Array3Writer(embeddings);
        let mut pending: Vec<PendingSplitEmbedding> = Vec::with_capacity(batch_size);
        let mut fbanks: Vec<Array2<f32>> = Vec::new();
        let mut tail_batches = 0usize;
        let mut active_items = 0usize;

        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let fbank = emb_model.compute_chunk_fbank(chunk_audio)?;
            fbanks.push(fbank);
            let mut current_fbank_idx = fbanks.len() - 1;

            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);

            for speaker_idx in 0..num_speakers {
                let Some(weights) = select_speaker_weights(
                    &chunk_segmentations,
                    &clean_masks,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    continue;
                };
                active_items += 1;
                pending.push(PendingSplitEmbedding {
                    chunk_idx,
                    speaker_idx,
                    fbank_idx: current_fbank_idx,
                    weights,
                });
                if pending.len() == batch_size {
                    flush_split(emb_model, &pending, &fbanks, &mut storage)?;
                    tail_batches += 1;
                    pending.clear();

                    if speaker_idx + 1 < num_speakers {
                        let kept_fbank = fbanks.swap_remove(current_fbank_idx);
                        fbanks.clear();
                        fbanks.push(kept_fbank);
                        current_fbank_idx = 0;
                    } else {
                        fbanks.clear();
                    }
                }
            }
        }

        if !pending.is_empty() {
            flush_split(emb_model, &pending, &fbanks, &mut storage)?;
            tail_batches += 1;
        }

        debug!(
            batches = tail_batches,
            active_items,
            total_items = self.0.shape()[0] * num_speakers,
            "Split embeddings complete (fbank+tail streaming)"
        );

        Ok(())
    }

    fn extract_multi_mask_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.multi_mask_batch_size();
        let num_speakers = self.0.shape()[2];
        let num_chunks = self.0.shape()[0];
        let min_num_samples = emb_model.min_num_samples();

        let mut storage = Array3Writer(embeddings);
        let mut fbank_buffer: Vec<Array2<f32>> = Vec::with_capacity(batch_size);
        let mut masks_buffer: Vec<Vec<f32>> = Vec::with_capacity(batch_size * num_speakers);
        let mut chunk_indices: Vec<usize> = Vec::with_capacity(batch_size);
        let mut batches = 0usize;

        for chunk_idx in 0..num_chunks {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let fbank = emb_model.compute_chunk_fbank(chunk_audio)?;
            fbank_buffer.push(fbank);
            chunk_indices.push(chunk_idx);

            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks_arr = clean_masks(&chunk_segmentations);

            for speaker_idx in 0..num_speakers {
                let Some(weights) = select_speaker_weights(
                    &chunk_segmentations,
                    &clean_masks_arr,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    masks_buffer.push(vec![0.0; 589]);
                    continue;
                };
                masks_buffer.push(weights);
            }

            if fbank_buffer.len() == batch_size {
                flush_multi_mask(
                    emb_model,
                    &fbank_buffer,
                    &masks_buffer,
                    &chunk_indices,
                    num_speakers,
                    &mut storage,
                )?;
                batches += 1;
                fbank_buffer.clear();
                masks_buffer.clear();
                chunk_indices.clear();
            }
        }

        if !fbank_buffer.is_empty() {
            flush_multi_mask(
                emb_model,
                &fbank_buffer,
                &masks_buffer,
                &chunk_indices,
                num_speakers,
                &mut storage,
            )?;
            batches += 1;
        }

        debug!(
            batches,
            total_chunks = num_chunks,
            "Multi-mask embeddings complete"
        );

        Ok(())
    }
}

fn read_source_chunk<S: AudioSource + ?Sized>(
    source: &S,
    layout: &ChunkLayout,
    chunk_idx: usize,
) -> Result<Vec<f32>, PipelineError> {
    let start = chunk_idx
        .checked_mul(layout.step_samples)
        .ok_or(AudioSourceError::Overflow)
        .map_err(crate::inference::segmentation::SegmentationError::from)?;
    if start >= source.sample_count() {
        return Ok(Vec::new());
    }
    let end = start
        .checked_add(layout.window_samples)
        .ok_or(AudioSourceError::Overflow)
        .map_err(crate::inference::segmentation::SegmentationError::from)?
        .min(source.sample_count());
    let mut audio = vec![0.0; end - start];
    source
        .read_exact_samples(start, &mut audio)
        .map_err(crate::inference::segmentation::SegmentationError::from)?;
    Ok(audio)
}

/// Lazily groups active speaker rows in the same chunk-major order as the slice-backed path.
/// This bounds scheduling metadata to one model batch regardless of recording duration.
struct ActiveSpeakerBatchIter<'a> {
    segmentations: &'a Array3<f32>,
    next_item: usize,
    batch_size: usize,
}

fn push_batch_item<T>(batch: &mut Vec<T>, item: T, capacity: usize) -> bool {
    batch.push(item);
    batch.len() == capacity
}

struct BatchRangeIter {
    next: usize,
    total: usize,
    capacity: usize,
}

impl BatchRangeIter {
    fn new(total: usize, capacity: usize) -> Self {
        Self {
            next: 0,
            total,
            capacity: capacity.max(1),
        }
    }
}

impl Iterator for BatchRangeIter {
    type Item = std::ops::Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.total {
            return None;
        }
        let start = self.next;
        self.next = self.next.saturating_add(self.capacity).min(self.total);
        Some(start..self.next)
    }
}

impl<'a> ActiveSpeakerBatchIter<'a> {
    fn new(segmentations: &'a Array3<f32>, batch_size: usize) -> Self {
        Self {
            segmentations,
            next_item: 0,
            batch_size: batch_size.max(1),
        }
    }
}

impl Iterator for ActiveSpeakerBatchIter<'_> {
    type Item = Vec<(usize, usize)>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunks = self.segmentations.shape()[0];
        let speakers = self.segmentations.shape()[2];
        let total = chunks.saturating_mul(speakers);
        let mut batch = Vec::with_capacity(self.batch_size);
        while self.next_item < total && batch.len() < self.batch_size {
            let item = self.next_item;
            self.next_item += 1;
            let chunk_idx = item / speakers;
            let speaker_idx = item % speakers;
            let mask = self.segmentations.slice(s![chunk_idx, .., speaker_idx]);
            if mask.iter().sum::<f32>() >= MIN_SPEAKER_ACTIVITY {
                batch.push((chunk_idx, speaker_idx));
            }
        }
        (!batch.is_empty()).then_some(batch)
    }
}

pub(in crate::pipeline) trait EmbeddingStorage {
    fn store(&mut self, chunk_idx: usize, speaker_idx: usize, embedding: &[f32]);
}

fn store_row<S: EmbeddingStorage>(
    storage: &mut S,
    chunk_idx: usize,
    speaker_idx: usize,
    row: ndarray::ArrayView1<'_, f32>,
) {
    if let Some(values) = row.as_slice() {
        storage.store(chunk_idx, speaker_idx, values);
        return;
    }

    let values = row.to_vec();
    storage.store(chunk_idx, speaker_idx, &values);
}

/// Writes embeddings into a pre-allocated Array3 by (chunk, speaker) index
pub(in crate::pipeline) struct Array3Writer<'a>(pub &'a mut Array3<f32>);

impl EmbeddingStorage for Array3Writer<'_> {
    fn store(&mut self, chunk_idx: usize, speaker_idx: usize, embedding: &[f32]) {
        self.0
            .slice_mut(s![chunk_idx, speaker_idx, ..])
            .assign(&ndarray::ArrayView1::from(embedding));
    }
}

pub(in crate::pipeline) fn flush_masked<S: EmbeddingStorage>(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingEmbedding<'_>],
    storage: &mut S,
) -> Result<(), PipelineError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| MaskedEmbeddingInput {
            audio: item.audio.as_slice(),
            mask: &item.mask,
            clean_mask: Some(&item.clean_mask),
        })
        .collect();
    let batch_embeddings = emb_model.embed_batch(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        store_row(
            storage,
            item.chunk_idx,
            item.speaker_idx,
            batch_embeddings.row(batch_idx),
        );
    }

    Ok(())
}

pub(in crate::pipeline) fn flush_source_masked<S: EmbeddingStorage>(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingSourceEmbedding],
    storage: &mut S,
) -> Result<(), PipelineError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| MaskedEmbeddingInput {
            audio: &item.audio,
            mask: &item.mask,
            clean_mask: Some(&item.clean_mask),
        })
        .collect();
    let batch_embeddings = emb_model.embed_batch(&batch_inputs)?;
    for (batch_idx, item) in pending.iter().enumerate() {
        store_row(
            storage,
            item.chunk_idx,
            item.speaker_idx,
            batch_embeddings.row(batch_idx),
        );
    }
    Ok(())
}

pub(in crate::pipeline) fn flush_split<S: EmbeddingStorage>(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingSplitEmbedding],
    fbanks: &[Array2<f32>],
    storage: &mut S,
) -> Result<(), PipelineError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| SplitTailInput {
            fbank: &fbanks[item.fbank_idx],
            weights: &item.weights,
        })
        .collect();
    let batch_embeddings = emb_model.embed_tail_batch_inputs(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        store_row(
            storage,
            item.chunk_idx,
            item.speaker_idx,
            batch_embeddings.row(batch_idx),
        );
    }

    Ok(())
}

pub(in crate::pipeline) fn flush_multi_mask<S: EmbeddingStorage>(
    emb_model: &mut EmbeddingModel,
    fbanks: &[Array2<f32>],
    masks: &[Vec<f32>],
    chunk_indices: &[usize],
    num_speakers: usize,
    storage: &mut S,
) -> Result<(), PipelineError> {
    let fbank_refs: Vec<_> = fbanks.iter().collect();
    let mask_refs: Vec<_> = masks.iter().map(|m| m.as_slice()).collect();
    let batch_embeddings = emb_model.embed_multi_mask_batch(&fbank_refs, &mask_refs)?;

    for (fbank_idx, &chunk_idx) in chunk_indices.iter().enumerate() {
        for speaker_idx in 0..num_speakers {
            let mask_idx = fbank_idx * num_speakers + speaker_idx;
            let is_active = masks[mask_idx].iter().any(|&v| v > 0.0);
            if !is_active {
                continue;
            }
            store_row(
                storage,
                chunk_idx,
                speaker_idx,
                batch_embeddings.row(mask_idx),
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod source_embedding_tests {
    use super::{ActiveSpeakerBatchIter, BatchRangeIter, read_source_chunk};
    use crate::audio_source::{AudioSource, AudioSourceError, SliceAudioSource};
    use crate::pipeline::types::ChunkLayout;
    use ndarray::Array3;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FailingSource {
        reads: AtomicUsize,
    }

    impl AudioSource for FailingSource {
        fn sample_count(&self) -> usize {
            64
        }

        fn read_exact_samples(
            &self,
            _start: usize,
            _destination: &mut [f32],
        ) -> Result<(), AudioSourceError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            Err(AudioSourceError::Cancelled)
        }
    }

    fn legacy_active_rows(segmentations: &Array3<f32>) -> Vec<(usize, usize)> {
        let mut rows = Vec::new();
        for chunk_idx in 0..segmentations.shape()[0] {
            for speaker_idx in 0..segmentations.shape()[2] {
                let mask = segmentations.slice(ndarray::s![chunk_idx, .., speaker_idx]);
                if mask.iter().sum::<f32>() >= crate::pipeline::MIN_SPEAKER_ACTIVITY {
                    rows.push((chunk_idx, speaker_idx));
                }
            }
        }
        rows
    }

    #[test]
    fn source_masked_batches_match_slice_order_and_keep_primary_tail_schedule() {
        let mut segmentations = Array3::<f32>::zeros((35, 10, 3));
        for chunk in 0..22 {
            for speaker in 0..3 {
                for frame in 0..10 {
                    segmentations[[chunk, frame, speaker]] = 1.0;
                }
            }
        }
        let expected = legacy_active_rows(&segmentations);
        let batches = ActiveSpeakerBatchIter::new(&segmentations, 64).collect::<Vec<_>>();
        let flattened = batches.iter().flatten().copied().collect::<Vec<_>>();
        assert_eq!(flattened, expected);
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![64, 2]
        );
        assert!(batches.iter().all(|batch| batch.len() <= 64));

        // The extraction loop writes each scheduled output row by its original location, so the
        // same schedule must produce the same result as the legacy chunk-major iteration.
        let mut batched_output = Array3::<f32>::zeros((35, 2, 3));
        for (batch_idx, batch) in batches.iter().enumerate() {
            for &(chunk, speaker) in batch {
                batched_output[[chunk, 0, speaker]] = (chunk * 3 + speaker + 1) as f32;
            }
            assert_eq!(batch_idx == 0, batch.len() == 64);
        }
        let mut legacy_output = Array3::<f32>::zeros((35, 2, 3));
        for (chunk, speaker) in expected {
            legacy_output[[chunk, 0, speaker]] = (chunk * 3 + speaker + 1) as f32;
        }
        assert_eq!(batched_output, legacy_output);
    }

    #[test]
    fn source_multimask_batches_retain_32_chunk_tail_and_bounded_scratch() {
        let chunks = 69;
        let capacity = 32;
        let ranges = BatchRangeIter::new(chunks, capacity).collect::<Vec<_>>();
        assert_eq!(
            ranges.iter().map(|r| r.len()).collect::<Vec<_>>(),
            vec![32, 32, 5]
        );
        assert!(ranges.iter().all(|range| range.len() <= capacity));
        assert_eq!(
            ranges
                .iter()
                .flat_map(|range| range.clone())
                .collect::<Vec<_>>(),
            (0..chunks).collect::<Vec<_>>()
        );
    }

    #[test]
    fn source_split_schedule_keeps_64_speaker_rows_across_chunk_boundaries() {
        let mut pending = Vec::with_capacity(64);
        let mut observed = Vec::new();
        let mut batch_lengths = Vec::new();
        for chunk in 0..22 {
            for speaker in 0..3 {
                if super::push_batch_item(&mut pending, (chunk, speaker), 64) {
                    batch_lengths.push(pending.len());
                    observed.extend(pending.drain(..));
                }
            }
        }
        if !pending.is_empty() {
            batch_lengths.push(pending.len());
            observed.extend(pending.drain(..));
        }
        assert_eq!(batch_lengths, vec![64, 2]);
        assert_eq!(
            observed,
            (0..22)
                .flat_map(|chunk| (0..3).map(move |speaker| (chunk, speaker)))
                .collect::<Vec<_>>()
        );
        assert!(batch_lengths.into_iter().all(|len| len <= 64));
    }

    #[test]
    fn source_chunk_reads_match_slice_tail_and_propagate_cancellation() {
        let samples = (0..23).map(|n| n as f32).collect::<Vec<_>>();
        let source = SliceAudioSource::new(&samples);
        let layout = ChunkLayout::new(0.25, 8, 10, 3);
        for chunk in 0..3 {
            assert_eq!(
                read_source_chunk(&source, &layout, chunk).unwrap(),
                layout.chunk_audio(&samples, chunk)
            );
        }
        let failing = FailingSource {
            reads: AtomicUsize::new(0),
        };
        assert!(matches!(
            read_source_chunk(&failing, &layout, 0),
            Err(crate::pipeline::PipelineError::Segmentation(
                crate::inference::segmentation::SegmentationError::AudioSource(
                    AudioSourceError::Cancelled
                )
            ))
        ));
        assert_eq!(failing.reads.load(Ordering::Relaxed), 1);
    }
}
