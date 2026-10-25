mod config;
pub use crate::binarize::BinarizeConfig;
pub use crate::clustering::ahc::AhcConfig;
pub use crate::clustering::vbx::VbxConfig;
pub(crate) use config::MIN_SPEAKER_ACTIVITY;
pub use config::{
    COREML_SEGMENTATION_STEP_SECONDS, CUDA_SEGMENTATION_STEP_SECONDS,
    FAST_SEGMENTATION_STEP_SECONDS, FRAME_DURATION_SECONDS, FRAME_STEP_SECONDS, PipelineConfig,
    ReconstructMethod, RuntimeConfig, SEGMENTATION_STEP_SECONDS, SEGMENTATION_WINDOW_SECONDS,
    segmentation_step_seconds,
};

mod types;
pub(crate) use types::FrameActivations;
pub use types::{
    BatchInput, ChunkEmbeddings, ChunkSpeakerClusters, DecodedSegmentations, DiarizationResult,
    DiscreteDiarization, InferenceArtifacts, PipelineError, SpeakerCountTrack,
};
use types::{ChunkLayout, EmbeddingPath, InferencePath, RawSegmentationWindows};
#[cfg(test)]
use types::{chunk_audio_raw, chunk_start_frames, total_output_frames};

pub(crate) mod clustering;
#[cfg(test)]
use clustering::mark_inactive_speakers;
pub(crate) use clustering::{clean_masks, select_speaker_weights, write_speaker_mask_to_slice};

mod concurrent;
use concurrent::{ConcurrentAudio, ConcurrentEmbeddingRunner};

mod post_inference;
pub use post_inference::post_inference;

#[cfg(feature = "coreml")]
mod chunk_embedding;

mod builder;
pub use builder::PipelineBuilder;

mod queued;
pub use queued::{
    QueueError, QueueReceiver, QueueReceiverIter, QueueSender, QueuedDiarizationJobId,
    QueuedDiarizationRequest, QueuedDiarizationResult,
};

#[cfg(test)]
mod tests;

use std::path::Path;

use ndarray::{Array2, Array3};
use tracing::{debug, trace};

use crate::audio_source::{AudioSource, AudioWindowPlan};
use crate::clustering::plda::PldaTransform;
use crate::inference::ExecutionMode;
use crate::inference::embedding::EmbeddingModel;
use crate::inference::segmentation::SegmentationModel;
use crate::powerset::PowersetMapping;

/// Shared run/query methods for both owned and borrowed pipeline facades.
/// Both structs provide `runner()`, `mode`, and `seg_model` with compatible types
macro_rules! pipeline_run_methods {
    () => {
        /// Diarize audio using default config, file id defaults to "file1"
        pub fn run(&mut self, audio: &[f32]) -> Result<DiarizationResult, PipelineError> {
            self.run_with_file_id(audio, "file1")
        }

        /// Diarize audio with a custom file identifier for RTTM output
        pub fn run_with_file_id(
            &mut self,
            audio: &[f32],
            file_id: &str,
        ) -> Result<DiarizationResult, PipelineError> {
            let config = self.pipeline_config();
            self.run_with_config(audio, file_id, &config)
        }

        /// Diarize audio with a custom file identifier and pipeline config
        pub fn run_with_config(
            &mut self,
            audio: &[f32],
            file_id: &str,
            config: &PipelineConfig,
        ) -> Result<DiarizationResult, PipelineError> {
            self.runner().run(audio, file_id, config)
        }

        /// Run only inference (segmentation + embedding), returning intermediate artifacts
        pub fn run_inference_only(
            &mut self,
            audio: &[f32],
        ) -> Result<InferenceArtifacts, PipelineError> {
            self.runner().run_inference(audio)
        }

        /// Run the standard sequential pipeline from a random-access audio source.
        ///
        /// Segmentation and embedding working buffers are bounded by their model batch sizes;
        /// global decoded segmentations and embeddings remain available for clustering.
        /// Accelerator modes that require concurrent whole-audio access are rejected.
        pub fn run_source<S: AudioSource + ?Sized>(
            &mut self,
            source: &S,
            file_id: &str,
        ) -> Result<DiarizationResult, PipelineError> {
            let config = self.pipeline_config();
            self.run_source_with_config(source, file_id, &config)
        }

        /// Run the standard source-backed pipeline with a custom clustering/reconstruction config.
        pub fn run_source_with_config<S: AudioSource + ?Sized>(
            &mut self,
            source: &S,
            file_id: &str,
            config: &PipelineConfig,
        ) -> Result<DiarizationResult, PipelineError> {
            self.runner().run_source(source, file_id, config)
        }

        /// Default pipeline config used by `run` and `run_batch`
        pub fn pipeline_config(&self) -> PipelineConfig {
            self.default_config.clone()
        }

        /// Diarize a batch of files, keeping all hardware busy across file boundaries
        pub fn run_batch(
            &mut self,
            files: &[BatchInput<'_>],
        ) -> Result<Vec<DiarizationResult>, PipelineError> {
            let config = self.pipeline_config();
            self.run_batch_with_config(files, &config)
        }

        /// Diarize a batch of files with custom config
        pub fn run_batch_with_config(
            &mut self,
            files: &[BatchInput<'_>],
            config: &PipelineConfig,
        ) -> Result<Vec<DiarizationResult>, PipelineError> {
            self.runner().run_batch(files, config)
        }

        /// Segmentation step size in seconds for the current execution mode
        pub fn segmentation_step(&self) -> f64 {
            self.seg_model.step_seconds()
        }
    };
}

/// Owned pipeline that manages its own model lifetimes
///
/// Construct directly for the default path, or use [`PipelineBuilder`] for custom config:
/// ```no_run
/// use speakrs::{ExecutionMode, OwnedDiarizationPipeline};
///
/// let mut pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::Cpu)?;
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
pub struct OwnedDiarizationPipeline {
    pub(crate) seg_model: SegmentationModel,
    pub(crate) emb_model: EmbeddingModel,
    pub(crate) plda: PldaTransform,
    pub(crate) powerset: PowersetMapping,
    pub(crate) default_config: PipelineConfig,
}

impl OwnedDiarizationPipeline {
    /// Load models from a local directory using default pipeline and runtime config
    pub fn from_dir(
        models_dir: impl Into<std::path::PathBuf>,
        mode: ExecutionMode,
    ) -> Result<Self, PipelineError> {
        PipelineBuilder::from_dir(models_dir, mode).build()
    }

    /// Build from a resolved [`ModelBundle`](crate::models::ModelBundle) using default config
    pub fn from_bundle(
        bundle: crate::models::ModelBundle,
        mode: ExecutionMode,
    ) -> Result<Self, PipelineError> {
        PipelineBuilder::from_bundle(bundle, mode).build()
    }

    /// Download models from HuggingFace and build with default config
    #[cfg(feature = "online")]
    pub fn from_pretrained(mode: ExecutionMode) -> Result<Self, PipelineError> {
        PipelineBuilder::from_pretrained(mode)?.build()
    }

    pipeline_run_methods!();

    /// Run post-inference (clustering + reconstruction) on pre-computed artifacts
    ///
    /// Does not need mutable model access, so it can run on a background
    /// thread while the next file's inference proceeds
    pub fn finish_post_inference(
        &self,
        artifacts: InferenceArtifacts,
        config: &PipelineConfig,
    ) -> Result<DiarizationResult, PipelineError> {
        post_inference(artifacts, config, &self.plda)
    }

    /// Convert into a background-processing queue
    pub fn into_queued(self) -> Result<(QueueSender, QueueReceiver), QueueError> {
        let config = self.default_config.clone();
        self.into_queued_with_config(config)
    }

    /// Convert into a background-processing queue with custom pipeline config
    pub fn into_queued_with_config(
        self,
        config: PipelineConfig,
    ) -> Result<(QueueSender, QueueReceiver), QueueError> {
        QueueSender::new(self, config)
    }
}

/// Borrowed pipeline for when you manage model lifetimes yourself
pub struct DiarizationPipeline<'a> {
    seg_model: &'a mut SegmentationModel,
    emb_model: &'a mut EmbeddingModel,
    plda: PldaTransform,
    powerset: PowersetMapping,
    default_config: PipelineConfig,
}

impl<'a> DiarizationPipeline<'a> {
    /// Build a pipeline from pre-loaded models and a PLDA parameters directory
    pub fn new(
        seg_model: &'a mut SegmentationModel,
        emb_model: &'a mut EmbeddingModel,
        models_dir: &Path,
    ) -> Result<Self, PipelineError> {
        let default_config = PipelineConfig::for_mode(seg_model.mode());
        Self::new_with_config(seg_model, emb_model, models_dir, default_config)
    }

    /// Build a pipeline from pre-loaded models with custom pipeline config
    pub fn new_with_config(
        seg_model: &'a mut SegmentationModel,
        emb_model: &'a mut EmbeddingModel,
        models_dir: &Path,
        default_config: PipelineConfig,
    ) -> Result<Self, PipelineError> {
        Ok(Self {
            seg_model,
            emb_model,
            plda: PldaTransform::from_dir(models_dir)?,
            powerset: PowersetMapping::new(3, 2),
            default_config,
        })
    }

    /// Default segmentation step in seconds (CPU mode)
    pub fn default_segmentation_step() -> f32 {
        segmentation_step_seconds(ExecutionMode::Cpu) as f32
    }

    pipeline_run_methods!();
}

impl OwnedDiarizationPipeline {
    fn runner(&mut self) -> PipelineRunner<'_> {
        PipelineRunner {
            seg_model: &mut self.seg_model,
            emb_model: &mut self.emb_model,
            plda: &self.plda,
            powerset: &self.powerset,
        }
    }
}

impl<'a> DiarizationPipeline<'a> {
    fn runner(&mut self) -> PipelineRunner<'_> {
        PipelineRunner {
            seg_model: self.seg_model,
            emb_model: self.emb_model,
            plda: &self.plda,
            powerset: &self.powerset,
        }
    }
}

struct PipelineRunner<'a> {
    seg_model: &'a mut SegmentationModel,
    emb_model: &'a mut EmbeddingModel,
    plda: &'a PldaTransform,
    powerset: &'a PowersetMapping,
}

impl<'a> PipelineRunner<'a> {
    fn run(
        &mut self,
        audio: &[f32],
        file_id: &str,
        config: &PipelineConfig,
    ) -> Result<DiarizationResult, PipelineError> {
        let run_start = std::time::Instant::now();
        let inference_artifacts = self.run_inference(audio)?;
        let inference_ms = run_start.elapsed().as_millis();
        let post_start = std::time::Instant::now();
        let result = self.run_post_inference(inference_artifacts, config)?;
        let post_ms = post_start.elapsed().as_millis();
        let total_ms = run_start.elapsed().as_millis();
        let audio_secs = audio.len() as f64 / 16_000.0;
        trace!(
            %file_id,
            inference_ms,
            post_ms,
            total_ms,
            audio_secs,
            "Pipeline complete",
        );
        Ok(result)
    }

    fn run_source<S: AudioSource + ?Sized>(
        &mut self,
        source: &S,
        file_id: &str,
        config: &PipelineConfig,
    ) -> Result<DiarizationResult, PipelineError> {
        let run_start = std::time::Instant::now();
        let inference_artifacts = self.run_inference_source(source)?;
        let inference_ms = run_start.elapsed().as_millis();
        let post_start = std::time::Instant::now();
        let result = self.run_post_inference(inference_artifacts, config)?;
        let post_ms = post_start.elapsed().as_millis();
        let total_ms = run_start.elapsed().as_millis();
        let audio_secs = source.sample_count() as f64 / 16_000.0;
        trace!(%file_id, inference_ms, post_ms, total_ms, audio_secs, "Source pipeline complete");
        Ok(result)
    }

    fn inference_path(&self) -> InferencePath {
        if matches!(
            self.seg_model.mode(),
            ExecutionMode::CoreMl
                | ExecutionMode::CoreMlFast
                | ExecutionMode::Cuda
                | ExecutionMode::CudaFast
                | ExecutionMode::MiGraphX
        ) {
            InferencePath::Concurrent
        } else {
            InferencePath::Sequential
        }
    }

    fn embedding_path(&self) -> EmbeddingPath {
        let path = if self.emb_model.prefers_multi_mask_path()
            && self.emb_model.multi_mask_batch_size() > 0
        {
            EmbeddingPath::MultiMask
        } else if self.emb_model.prefers_chunk_embedding_path()
            && self.emb_model.split_primary_batch_size() > 0
        {
            EmbeddingPath::Split
        } else {
            EmbeddingPath::Masked
        };
        debug!(?path, "Embedding path selected");
        path
    }

    fn run_batch(
        &mut self,
        files: &[BatchInput<'_>],
        config: &PipelineConfig,
    ) -> Result<Vec<DiarizationResult>, PipelineError> {
        if files.is_empty() {
            return Ok(Vec::new());
        }

        // try batch chunk embedding for the concurrent path before falling back
        #[cfg(feature = "coreml")]
        if matches!(self.inference_path(), InferencePath::Concurrent)
            && let Some(results) = chunk_embedding::try_batch_chunk_embedding(
                self.seg_model,
                self.emb_model,
                self.powerset,
                self.plda,
                files,
                config,
            )?
        {
            return Ok(results);
        }

        // fallback: per-file sequential with post-inference overlap
        files
            .iter()
            .map(|f| self.run(f.audio, f.file_id, config))
            .collect()
    }

    fn run_inference(&mut self, audio: &[f32]) -> Result<InferenceArtifacts, PipelineError> {
        match self.inference_path() {
            InferencePath::Sequential => self.run_sequential_inference(audio),
            InferencePath::Concurrent => {
                #[cfg(feature = "coreml")]
                if let Some(result) = chunk_embedding::try_chunk_embedding(
                    self.seg_model,
                    self.emb_model,
                    self.powerset,
                    audio,
                )? {
                    return Ok(result);
                }
                self.run_concurrent_inference(audio)
            }
        }
    }

    fn run_sequential_inference(
        &mut self,
        audio: &[f32],
    ) -> Result<InferenceArtifacts, PipelineError> {
        let raw_windows = RawSegmentationWindows(self.seg_model.run(audio)?);
        debug!(windows = raw_windows.0.len(), "Segmentation complete");

        let segmentations = raw_windows.decode(self.powerset);
        let layout = ChunkLayout::new(
            self.seg_model.step_seconds(),
            self.seg_model.step_samples(),
            self.seg_model.window_samples(),
            segmentations.nchunks(),
        );
        let embeddings = segmentations.extract_embeddings(
            audio,
            self.emb_model,
            &layout,
            self.embedding_path(),
        )?;

        debug!(
            chunks = segmentations.nchunks(),
            speakers = segmentations.num_speakers(),
            "Embeddings complete"
        );

        Ok(InferenceArtifacts {
            layout,
            segmentations,
            embeddings,
        })
    }

    fn run_inference_source<S: AudioSource + ?Sized>(
        &mut self,
        source: &S,
    ) -> Result<InferenceArtifacts, PipelineError> {
        if !matches!(self.inference_path(), InferencePath::Sequential) {
            return self.run_concurrent_inference_source(source);
        }

        let window_samples = self.seg_model.window_samples();
        let step_samples = self.seg_model.step_samples();
        let plan = AudioWindowPlan::new(source.sample_count(), window_samples, step_samples)
            .map_err(crate::inference::segmentation::SegmentationError::from)?;
        let expected_windows = plan.window_count();
        let seg_model = &mut self.seg_model;
        let powerset = self.powerset;
        let collected = collect_source_decoded_windows(
            expected_windows,
            |tx| seg_model.run_streaming_source(source, tx),
            |window| powerset.hard_decode(window),
        )?;
        debug!(
            windows = expected_windows,
            max_queued_raw_windows = collected.max_queued_raw_windows,
            "Source segmentation complete"
        );

        let segmentations = collected.segmentations;
        let layout = ChunkLayout::new(
            self.seg_model.step_seconds(),
            step_samples,
            window_samples,
            segmentations.nchunks(),
        );
        let embeddings = segmentations.extract_embeddings_source(
            source,
            self.emb_model,
            &layout,
            self.embedding_path(),
        )?;
        debug!(
            chunks = segmentations.nchunks(),
            speakers = segmentations.num_speakers(),
            "Source embeddings complete"
        );
        Ok(InferenceArtifacts {
            layout,
            segmentations,
            embeddings,
        })
    }

    fn run_concurrent_inference(
        &mut self,
        audio: &[f32],
    ) -> Result<InferenceArtifacts, PipelineError> {
        let layout = ChunkLayout::without_frame_extent(
            self.seg_model.step_seconds(),
            self.seg_model.step_samples(),
            self.seg_model.window_samples(),
        );
        let concurrent_embedding_runner = ConcurrentEmbeddingRunner {
            powerset: self.powerset,
            audio: ConcurrentAudio::Slice(audio),
            step_samples: layout.step_samples,
            window_samples: layout.window_samples,
            num_speakers: 3,
        };
        let embedding_path = self.embedding_path();
        let batch_size = match embedding_path {
            EmbeddingPath::MultiMask => self.emb_model.multi_mask_batch_size(),
            EmbeddingPath::Split => self.emb_model.split_primary_batch_size(),
            EmbeddingPath::Masked => self.emb_model.primary_batch_size(),
        };
        let min_num_samples = self.emb_model.min_num_samples();
        let (tx, rx) = crossbeam_channel::bounded::<Array2<f32>>(64);

        let inference_start = std::time::Instant::now();
        let use_parallel_seg = matches!(
            self.seg_model.mode(),
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast
        );
        let (segmentation_result, embedding_result) = std::thread::scope(|scope| {
            let segmentation_handle = if use_parallel_seg {
                #[cfg(feature = "coreml")]
                {
                    scope.spawn(|| self.seg_model.run_streaming_parallel(audio, tx, 4, None))
                }
                #[cfg(not(feature = "coreml"))]
                {
                    scope.spawn(|| self.seg_model.run_streaming(audio, tx))
                }
            } else {
                scope.spawn(|| self.seg_model.run_streaming(audio, tx))
            };

            let embedding_result = match embedding_path {
                EmbeddingPath::MultiMask => concurrent_embedding_runner.run_multi_mask(
                    rx,
                    self.emb_model,
                    batch_size,
                    min_num_samples,
                ),
                EmbeddingPath::Split => concurrent_embedding_runner.run_split(
                    rx,
                    self.emb_model,
                    batch_size,
                    min_num_samples,
                ),
                EmbeddingPath::Masked => {
                    concurrent_embedding_runner.run_masked(rx, self.emb_model, batch_size)
                }
            };

            let segmentation_result =
                segmentation_handle
                    .join()
                    .map_err(|_| PipelineError::WorkerPanic {
                        worker: "concurrent segmentation".to_owned(),
                    });
            (segmentation_result, embedding_result)
        });
        let inference_elapsed = inference_start.elapsed();

        segmentation_result??;

        let concurrent_result = embedding_result?;
        if concurrent_result.is_empty() {
            return Ok(Self::empty_inference_artifacts(layout));
        }

        let num_chunks = concurrent_result.num_chunks;
        let layout = layout.with_num_chunks(num_chunks);
        debug!(
            chunks = concurrent_result.segmentations.shape()[0],
            speakers = concurrent_result.segmentations.shape()[2],
            inference_ms = inference_elapsed.as_millis(),
            embedding_path = ?embedding_path,
            "Concurrent seg+emb complete"
        );

        Ok(InferenceArtifacts {
            layout,
            segmentations: DecodedSegmentations(concurrent_result.segmentations),
            embeddings: ChunkEmbeddings(concurrent_result.embeddings),
        })
    }

    fn run_concurrent_inference_source<S: AudioSource + ?Sized>(
        &mut self,
        source: &S,
    ) -> Result<InferenceArtifacts, PipelineError> {
        #[cfg(feature = "coreml")]
        if let Some(result) = chunk_embedding::try_chunk_embedding_source(
            self.seg_model,
            self.emb_model,
            self.powerset,
            source,
        )? {
            return Ok(result);
        }

        let read_source =
            |start, destination: &mut [f32]| source.read_exact_samples(start, destination);
        let layout = ChunkLayout::without_frame_extent(
            self.seg_model.step_seconds(),
            self.seg_model.step_samples(),
            self.seg_model.window_samples(),
        );
        let concurrent_embedding_runner = ConcurrentEmbeddingRunner {
            powerset: self.powerset,
            audio: ConcurrentAudio::Source {
                sample_count: source.sample_count(),
                read_exact: &read_source,
            },
            step_samples: layout.step_samples,
            window_samples: layout.window_samples,
            num_speakers: 3,
        };
        let embedding_path = self.embedding_path();
        let batch_size = match embedding_path {
            EmbeddingPath::MultiMask => self.emb_model.multi_mask_batch_size(),
            EmbeddingPath::Split => self.emb_model.split_primary_batch_size(),
            EmbeddingPath::Masked => self.emb_model.primary_batch_size(),
        };
        let min_num_samples = self.emb_model.min_num_samples();
        let (tx, rx) = crossbeam_channel::bounded::<Array2<f32>>(64);
        let inference_start = std::time::Instant::now();
        let use_parallel_seg = matches!(
            self.seg_model.mode(),
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast
        );
        let (segmentation_result, embedding_result) = std::thread::scope(|scope| {
            let segmentation_handle = if use_parallel_seg {
                #[cfg(feature = "coreml")]
                {
                    scope.spawn(|| {
                        self.seg_model
                            .run_streaming_parallel_source(source, tx, 4, None)
                    })
                }
                #[cfg(not(feature = "coreml"))]
                {
                    scope.spawn(|| self.seg_model.run_streaming_source(source, tx))
                }
            } else {
                scope.spawn(|| self.seg_model.run_streaming_source(source, tx))
            };

            let embedding_result = match embedding_path {
                EmbeddingPath::MultiMask => concurrent_embedding_runner.run_multi_mask(
                    rx,
                    self.emb_model,
                    batch_size,
                    min_num_samples,
                ),
                EmbeddingPath::Split => concurrent_embedding_runner.run_split(
                    rx,
                    self.emb_model,
                    batch_size,
                    min_num_samples,
                ),
                EmbeddingPath::Masked => {
                    concurrent_embedding_runner.run_masked(rx, self.emb_model, batch_size)
                }
            };
            let segmentation_result =
                segmentation_handle
                    .join()
                    .map_err(|_| PipelineError::WorkerPanic {
                        worker: "source concurrent segmentation".to_owned(),
                    });
            (segmentation_result, embedding_result)
        });
        let inference_elapsed = inference_start.elapsed();
        segmentation_result??;
        let concurrent_result = embedding_result?;
        if concurrent_result.is_empty() {
            return Ok(Self::empty_inference_artifacts(layout));
        }
        let num_chunks = concurrent_result.num_chunks;
        let layout = layout.with_num_chunks(num_chunks);
        debug!(
            chunks = concurrent_result.segmentations.shape()[0],
            speakers = concurrent_result.segmentations.shape()[2],
            inference_ms = inference_elapsed.as_millis(),
            embedding_path = ?embedding_path,
            "Source concurrent seg+emb complete"
        );
        Ok(InferenceArtifacts {
            layout,
            segmentations: DecodedSegmentations(concurrent_result.segmentations),
            embeddings: ChunkEmbeddings(concurrent_result.embeddings),
        })
    }

    fn run_post_inference(
        &mut self,
        inference_artifacts: InferenceArtifacts,
        config: &PipelineConfig,
    ) -> Result<DiarizationResult, PipelineError> {
        post_inference(inference_artifacts, config, self.plda)
    }

    fn empty_inference_artifacts(layout: ChunkLayout) -> InferenceArtifacts {
        InferenceArtifacts {
            layout: layout.with_num_chunks(0),
            segmentations: DecodedSegmentations(Array3::zeros((0, 0, 0))),
            embeddings: ChunkEmbeddings(Array3::zeros((0, 0, 0))),
        }
    }
}

struct CollectedSourceSegmentations {
    segmentations: DecodedSegmentations,
    max_queued_raw_windows: usize,
}

fn collect_source_decoded_windows<F, D>(
    expected_windows: usize,
    producer: F,
    mut decode: D,
) -> Result<CollectedSourceSegmentations, PipelineError>
where
    F: FnOnce(
            crossbeam_channel::Sender<Array2<f32>>,
        ) -> Result<usize, crate::inference::segmentation::SegmentationError>
        + Send,
    D: FnMut(&Array2<f32>) -> Array2<f32>,
{
    const RAW_WINDOW_CHANNEL_CAPACITY: usize = 4;
    let (tx, rx) = crossbeam_channel::bounded(RAW_WINDOW_CHANNEL_CAPACITY);
    std::thread::scope(|scope| {
        let handle = scope.spawn(move || producer(tx));
        let mut received = 0usize;
        let mut decoded: Option<Array3<f32>> = None;
        let mut decoded_shape = None;
        let mut sink_error = None;
        let mut max_queued_raw_windows = 0usize;
        while let Ok(raw_window) = rx.recv() {
            max_queued_raw_windows = max_queued_raw_windows.max(rx.len());
            if received < expected_windows && sink_error.is_none() {
                let decoded_window = decode(&raw_window);
                let shape = decoded_window.dim();
                match decoded_shape {
                    None => {
                        decoded = Some(Array3::zeros((expected_windows, shape.0, shape.1)));
                        decoded_shape = Some(shape);
                    }
                    Some(expected_shape) if expected_shape != shape => {
                        sink_error = Some(PipelineError::Invariant(format!(
                            "source segmentation window shape changed from {expected_shape:?} to {shape:?}"
                        )));
                    }
                    Some(_) => {}
                }
                if sink_error.is_none() {
                    if let Some(output) = decoded.as_mut() {
                        output
                            .slice_mut(ndarray::s![received, .., ..])
                            .assign(&decoded_window);
                    } else {
                        sink_error = Some(PipelineError::Invariant(
                            "source decoded sink was not allocated".into(),
                        ));
                    }
                }
            } else if received >= expected_windows && sink_error.is_none() {
                sink_error = Some(PipelineError::Invariant(format!(
                    "source segmentation sent more than the expected {expected_windows} windows"
                )));
            }
            received = received.saturating_add(1);
            // raw_window and decoded_window are dropped on each iteration; only the final decoded
            // annotations survive this bounded channel drain.
        }
        let reported = handle.join().map_err(|_| PipelineError::WorkerPanic {
            worker: "source segmentation".into(),
        })??;
        if let Some(error) = sink_error {
            return Err(error);
        }
        if reported != expected_windows || received != expected_windows {
            return Err(PipelineError::Invariant(format!(
                "source segmentation produced {reported} windows and {received} outputs; expected {expected_windows}"
            )));
        }
        let segmentations = decoded.unwrap_or_else(|| Array3::zeros((0, 0, 0)));
        Ok(CollectedSourceSegmentations {
            segmentations: DecodedSegmentations(segmentations),
            max_queued_raw_windows,
        })
    })
}

#[cfg(test)]
mod source_pipeline_tests {
    use super::collect_source_decoded_windows;
    use crate::audio_source::AudioSourceError;
    use crate::inference::segmentation::SegmentationError;
    use crate::pipeline::PipelineError;
    use ndarray::Array2;

    #[test]
    fn bounded_source_collector_preserves_window_order_and_joins_producer() {
        let collected = collect_source_decoded_windows(
            9,
            |tx| {
                for index in 0..9 {
                    tx.send(Array2::from_elem((1, 1), index as f32))?;
                }
                Ok(9)
            },
            |window| window.mapv(|value| value + 1.0),
        )
        .unwrap();
        assert_eq!(collected.segmentations.nchunks(), 9);
        for index in 0..9 {
            assert_eq!(collected.segmentations.0[[index, 0, 0]], index as f32 + 1.0);
        }
        assert!(collected.max_queued_raw_windows <= 4);
    }

    #[test]
    fn bounded_source_collector_joins_after_reader_cancellation() {
        let result = collect_source_decoded_windows(
            9,
            |tx| {
                tx.send(Array2::zeros((1, 1)))?;
                Err(SegmentationError::AudioSource(AudioSourceError::Cancelled))
            },
            |window| window.clone(),
        );
        assert!(matches!(
            result,
            Err(PipelineError::Segmentation(SegmentationError::AudioSource(
                AudioSourceError::Cancelled
            )))
        ));
    }

    #[test]
    fn bounded_source_collector_rejects_incomplete_or_misreported_output() {
        let result = collect_source_decoded_windows(
            2,
            |tx| {
                tx.send(Array2::zeros((1, 1)))?;
                Ok(2)
            },
            |window| window.clone(),
        );
        assert!(matches!(result, Err(PipelineError::Invariant(_))));
    }

    #[test]
    fn bounded_source_collector_reports_worker_panic_after_join() {
        let result = collect_source_decoded_windows(
            1,
            |_tx| -> Result<usize, SegmentationError> { panic!("test producer panic") },
            |window| window.clone(),
        );
        assert!(matches!(
            result,
            Err(PipelineError::WorkerPanic { worker }) if worker == "source segmentation"
        ));
    }

    #[test]
    fn decoded_source_sink_rejects_unstable_window_shapes() {
        let result = collect_source_decoded_windows(
            2,
            |tx| {
                tx.send(Array2::zeros((1, 1)))?;
                tx.send(Array2::zeros((2, 1)))?;
                Ok(2)
            },
            |window| window.clone(),
        );
        assert!(matches!(result, Err(PipelineError::Invariant(_))));
    }
}
