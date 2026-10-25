use ndarray::ArrayViewD;
use ndarray::{Array1, Array2, Array3, array, s};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::{Path, PathBuf};

use super::*;
#[cfg(any(feature = "coreml", feature = "cuda", feature = "migraphx"))]
use crate::inference::ExecutionMode;
use crate::inference::{DynamicRuntimeError, ModelLoadError, OrtRuntimeError};
#[cfg(any(feature = "cuda", feature = "migraphx"))]
use crate::audio_source::{AudioSource, AudioSourceError, SliceAudioSource};
#[cfg(any(feature = "cuda", feature = "migraphx"))]
use std::sync::atomic::{AtomicUsize, Ordering};

// --- test helpers ---

#[allow(dead_code)]
fn decode_windows(raw_windows: Vec<Array2<f32>>, powerset: &PowersetMapping) -> Array3<f32> {
    RawSegmentationWindows(raw_windows).decode(powerset).0
}

fn extract_embeddings(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
    segmentations: &Array3<f32>,
) -> Result<Array3<f32>, PipelineError> {
    let decoded_segmentations = DecodedSegmentations(segmentations.clone());
    let layout = ChunkLayout::new(
        seg_model.step_seconds(),
        seg_model.step_samples(),
        seg_model.window_samples(),
        decoded_segmentations.nchunks(),
    );
    let embedding_path = if emb_model.prefers_multi_mask_path()
        && emb_model.multi_mask_batch_size() > 0
    {
        EmbeddingPath::MultiMask
    } else if emb_model.prefers_chunk_embedding_path() && emb_model.split_primary_batch_size() > 0 {
        EmbeddingPath::Split
    } else {
        EmbeddingPath::Masked
    };
    decoded_segmentations
        .extract_embeddings(audio, emb_model, &layout, embedding_path)
        .map(|chunk_embeddings| chunk_embeddings.0)
}

fn filter_embeddings(
    segmentations: &Array3<f32>,
    embeddings: &Array3<f32>,
) -> (Array2<f32>, Vec<usize>, Vec<usize>) {
    let num_frames = segmentations.shape()[1] as f32;
    let mut filtered = Vec::new();
    let mut chunk_indices = Vec::new();
    let mut speaker_indices = Vec::new();

    for chunk_idx in 0..segmentations.shape()[0] {
        let single_active: Vec<bool> = segmentations
            .slice(s![chunk_idx, .., ..])
            .rows()
            .into_iter()
            .map(|row| (row.iter().copied().sum::<f32>() - 1.0).abs() < 1e-6)
            .collect();
        for speaker_idx in 0..segmentations.shape()[2] {
            let clean_frames = segmentations
                .slice(s![chunk_idx, .., speaker_idx])
                .iter()
                .zip(single_active.iter())
                .filter_map(|(value, is_single_active)| is_single_active.then_some(*value))
                .sum::<f32>();
            let embedding = embeddings.slice(s![chunk_idx, speaker_idx, ..]);
            let valid_embedding = embedding.iter().all(|value| value.is_finite());
            if valid_embedding && clean_frames >= 0.2 * num_frames {
                filtered.extend(embedding.iter());
                chunk_indices.push(chunk_idx);
                speaker_indices.push(speaker_idx);
            }
        }
    }

    let filtered_embeddings =
        Array2::from_shape_vec((chunk_indices.len(), embeddings.shape()[2]), filtered).unwrap();
    (filtered_embeddings, chunk_indices, speaker_indices)
}

#[allow(dead_code)]
fn chunk_audio<'a>(audio: &'a [f32], seg_model: &SegmentationModel, chunk_idx: usize) -> &'a [f32] {
    chunk_audio_raw(
        audio,
        seg_model.step_samples(),
        seg_model.window_samples(),
        chunk_idx,
    )
}

fn assign_embeddings(
    segmentations: &Array3<f32>,
    embeddings: &Array3<f32>,
    centroids: &Array2<f32>,
) -> Array2<i32> {
    super::clustering::assign_chunk_embeddings(
        &DecodedSegmentations(segmentations.clone()),
        &ChunkEmbeddings(embeddings.clone()),
        centroids,
    )
}

fn weighted_centroids(
    train_embeddings: &Array2<f32>,
    gamma: &Array2<f32>,
    kept_speakers: &[usize],
) -> Array2<f32> {
    super::clustering::weighted_centroids(train_embeddings, gamma, kept_speakers)
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn models_dir() -> PathBuf {
    fixture_path("models")
}

fn load_fixture_array1<T>(name: &str) -> Array1<T>
where
    Array1<T>: ReadNpyExt,
{
    Array1::read_npy(File::open(fixture_path(name)).unwrap()).unwrap()
}

fn load_fixture_array2<T>(name: &str) -> Array2<T>
where
    Array2<T>: ReadNpyExt,
{
    Array2::read_npy(File::open(fixture_path(name)).unwrap()).unwrap()
}

fn load_fixture_array3<T>(name: &str) -> Array3<T>
where
    Array3<T>: ReadNpyExt,
{
    Array3::read_npy(File::open(fixture_path(name)).unwrap()).unwrap()
}

fn load_test_audio() -> (Vec<f32>, u32) {
    load_wav_samples(&fixture_path("test.wav"))
}

struct TestAudio {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl TestAudio {
    fn load() -> Self {
        let (samples, sample_rate) = load_test_audio();
        Self {
            samples,
            sample_rate,
        }
    }

    fn samples(&self) -> &[f32] {
        &self.samples
    }

    fn assert_16khz(&self) {
        assert_eq!(self.sample_rate, 16_000);
    }
}

struct PipelineTestHarness {
    models_dir: PathBuf,
    audio: TestAudio,
}

impl PipelineTestHarness {
    fn load() -> Self {
        Self {
            models_dir: models_dir(),
            audio: TestAudio::load(),
        }
    }

    fn audio(&self) -> &[f32] {
        self.audio.assert_16khz();
        self.audio.samples()
    }

    fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    fn segmentation_model_path(&self) -> PathBuf {
        self.models_dir.join("segmentation-3.0.onnx")
    }

    fn embedding_model_path(&self) -> PathBuf {
        self.models_dir.join("wespeaker-voxceleb-resnet34.onnx")
    }

    fn cpu_seg_model(&self) -> Option<SegmentationModel> {
        load_model_or_skip(SegmentationModel::new(
            self.segmentation_model_path(),
            SEGMENTATION_STEP_SECONDS as f32,
        ))
    }

    fn cpu_emb_model(&self) -> Option<EmbeddingModel> {
        load_model_or_skip(EmbeddingModel::new(self.embedding_model_path()))
    }

    fn cpu_pipeline(&self) -> Option<OwnedDiarizationPipeline> {
        build_pipeline_or_skip(
            PipelineBuilder::from_dir(self.models_dir(), ExecutionMode::Cpu).build(),
        )
    }

    #[cfg(feature = "coreml")]
    fn coreml_seg_model(&self) -> Option<SegmentationModel> {
        load_model_or_skip(SegmentationModel::with_mode(
            self.segmentation_model_path(),
            SEGMENTATION_STEP_SECONDS as f32,
            ExecutionMode::CoreMl,
        ))
    }

    #[cfg(feature = "coreml")]
    fn coreml_emb_model(&self) -> Option<EmbeddingModel> {
        load_model_or_skip(EmbeddingModel::with_mode(
            self.embedding_model_path(),
            ExecutionMode::CoreMl,
        ))
    }

    #[cfg(feature = "coreml")]
    fn coreml_pipeline(&self) -> Option<OwnedDiarizationPipeline> {
        build_pipeline_or_skip(
            PipelineBuilder::from_dir(self.models_dir(), ExecutionMode::CoreMl).build(),
        )
    }
}

fn custom_pipeline_config() -> PipelineConfig {
    PipelineConfig {
        merge_gap: 0.75,
        speaker_keep_threshold: 0.25,
        reconstruct_method: ReconstructMethod::Standard,
        ..PipelineConfig::default()
    }
}

fn load_wav_samples(path: &Path) -> (Vec<f32>, u32) {
    let data = std::fs::read(path).unwrap();
    let sample_rate = u32::from_le_bytes(data[24..28].try_into().unwrap());
    let bits_per_sample = u16::from_le_bytes(data[34..36].try_into().unwrap());
    assert_eq!(bits_per_sample, 16);

    let mut pos = 12;
    while pos + 8 < data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if chunk_id == b"data" {
            let samples = data[pos + 8..pos + 8 + chunk_size]
                .chunks_exact(2)
                .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0)
                .collect();
            return (samples, sample_rate);
        }
        pos += 8 + chunk_size;
    }

    panic!("no data chunk found in WAV");
}

fn load_model_or_skip<T>(result: Result<T, ModelLoadError>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(ModelLoadError::Runtime(OrtRuntimeError::Dynamic(DynamicRuntimeError::Missing {
            ..
        }))) if cfg!(feature = "load-dynamic") => {
            eprintln!("skipping model-loading test because ORT_DYLIB_PATH is not configured");
            None
        }
        Err(error) => panic!("failed to load model: {error}"),
    }
}

fn build_pipeline_or_skip<T>(result: Result<T, PipelineError>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(PipelineError::ModelLoad(ModelLoadError::Runtime(OrtRuntimeError::Dynamic(
            DynamicRuntimeError::Missing { .. },
        )))) if cfg!(feature = "load-dynamic") => {
            eprintln!("skipping pipeline test because ORT_DYLIB_PATH is not configured");
            None
        }
        Err(error) => panic!("failed to build pipeline: {error}"),
    }
}

fn assert_embedding_tensor_close(actual: &Array3<f32>, expected: &Array3<f32>, epsilon: f32) {
    for chunk_idx in 0..actual.shape()[0] {
        for speaker_idx in 0..actual.shape()[1] {
            for dim_idx in 0..actual.shape()[2] {
                let lhs = actual[[chunk_idx, speaker_idx, dim_idx]];
                let rhs = expected[[chunk_idx, speaker_idx, dim_idx]];
                if (lhs - rhs).abs() > epsilon || lhs.is_nan() != rhs.is_nan() {
                    panic!(
                        "chunk={chunk_idx} speaker={speaker_idx} dim={dim_idx} left={lhs} right={rhs}"
                    );
                }
            }
        }
    }
}

fn compare_f32_arrays(
    field: &str,
    actual: ArrayViewD<'_, f32>,
    expected: ArrayViewD<'_, f32>,
    absolute_tolerance: f32,
    relative_tolerance: f32,
    allow_matching_nan: bool,
) -> Result<(), String> {
    if actual.shape() != expected.shape() {
        return Err(format!(
            "{field} shape differs: {:?} vs {:?}",
            actual.shape(),
            expected.shape()
        ));
    }
    if actual.is_empty() {
        return Err(format!("{field} unexpectedly has no values"));
    }

    let mut finite_values = 0;
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if actual.is_nan() && expected.is_nan() && allow_matching_nan {
            continue;
        }
        if !actual.is_finite() || !expected.is_finite() {
            return Err(format!(
                "{field}[{index}] has unmatched NaN or infinity: slice={actual}, source={expected}"
            ));
        }
        finite_values += 1;
        let tolerance = absolute_tolerance + relative_tolerance * expected.abs();
        if (actual - expected).abs() > tolerance {
            return Err(format!(
                "{field}[{index}] differs: slice={actual}, source={expected}, tolerance={tolerance}"
            ));
        }
    }
    if finite_values == 0 {
        return Err(format!("{field} contains no finite values"));
    }
    Ok(())
}

fn assert_f32_arrays_close(
    field: &str,
    actual: ArrayViewD<'_, f32>,
    expected: ArrayViewD<'_, f32>,
    absolute_tolerance: f32,
    relative_tolerance: f32,
    allow_matching_nan: bool,
) {
    if let Err(error) = compare_f32_arrays(
        field,
        actual,
        expected,
        absolute_tolerance,
        relative_tolerance,
        allow_matching_nan,
    ) {
        panic!("{error}");
    }
}

#[cfg(feature = "coreml")]
fn assert_coreml_pipeline_results_match(
    slice_result: &DiarizationResult,
    source_result: &DiarizationResult,
    file_id: &str,
) {
    assert!(
        !slice_result.segments.is_empty(),
        "fixture must produce diarized speech"
    );
    assert_eq!(slice_result.speaker_count.0, source_result.speaker_count.0);
    assert_eq!(slice_result.hard_clusters.0, source_result.hard_clusters.0);
    assert_eq!(slice_result.segments, source_result.segments);
    assert_eq!(slice_result.rttm(file_id), source_result.rttm(file_id));
    assert_f32_arrays_close(
        "CoreML segmentations",
        slice_result.segmentations.0.view().into_dyn(),
        source_result.segmentations.0.view().into_dyn(),
        1e-5,
        1e-5,
        false,
    );
    assert_f32_arrays_close(
        "CoreML embeddings",
        slice_result.embeddings.0.view().into_dyn(),
        source_result.embeddings.0.view().into_dyn(),
        1e-5,
        1e-5,
        true,
    );
    assert_eq!(
        slice_result.discrete_diarization.0,
        source_result.discrete_diarization.0
    );
}

#[cfg(feature = "coreml")]
fn assert_segmentation_tensor_matches(actual: &Array3<f32>, expected: &Array3<f32>) {
    for chunk_idx in 0..actual.shape()[0] {
        for frame_idx in 0..actual.shape()[1] {
            for speaker_idx in 0..actual.shape()[2] {
                let lhs = actual[[chunk_idx, frame_idx, speaker_idx]];
                let rhs = expected[[chunk_idx, frame_idx, speaker_idx]];
                if lhs != rhs {
                    panic!(
                        "chunk={chunk_idx} frame={frame_idx} speaker={speaker_idx} left={lhs} right={rhs}"
                    );
                }
            }
        }
    }
}

// --- tests ---

#[test]
fn chunk_start_frames_match_pyannote_rounding() {
    assert_eq!(
        chunk_start_frames(4, SEGMENTATION_STEP_SECONDS),
        vec![0, 59, 119, 178]
    );
}

#[test]
fn total_output_frames_match_pyannote_aggregate_extent() {
    assert_eq!(total_output_frames(4, SEGMENTATION_STEP_SECONDS), 771);
}

#[test]
fn best_assignment_handles_more_speakers_than_clusters() {
    let scores = array![[0.9, 0.1], [0.8, 0.2], [0.1, 0.95]];
    let assignment = super::clustering::best_assignment(&scores, &[0, 1, 2], 2);
    assert_eq!(assignment.len(), 2);
    assert!(assignment.contains(&(0, 0)) || assignment.contains(&(1, 0)));
    assert!(assignment.contains(&(2, 1)));
}

#[test]
fn filter_embeddings_matches_python_fixture() {
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let embeddings: Array3<f32> = load_fixture_array3("pipeline_embeddings_data.npy");
    let expected_train_embeddings: Array2<f32> =
        load_fixture_array2("pipeline_train_embeddings.npy");
    let expected_chunk_idx: Array1<i64> = load_fixture_array1("pipeline_train_chunk_idx.npy");
    let expected_speaker_idx: Array1<i64> = load_fixture_array1("pipeline_train_speaker_idx.npy");

    let (train_embeddings, chunk_idx, speaker_idx) = filter_embeddings(&segmentations, &embeddings);

    assert_eq!(chunk_idx.len(), expected_chunk_idx.len());
    assert_eq!(speaker_idx.len(), expected_speaker_idx.len());
    for (lhs, rhs) in chunk_idx.iter().zip(expected_chunk_idx.iter()) {
        assert_eq!(*lhs as i64, *rhs);
    }
    for (lhs, rhs) in speaker_idx.iter().zip(expected_speaker_idx.iter()) {
        assert_eq!(*lhs as i64, *rhs);
    }
    for (lhs, rhs) in train_embeddings
        .iter()
        .zip(expected_train_embeddings.iter())
    {
        approx::assert_abs_diff_eq!(*lhs, *rhs, epsilon = 1e-5);
    }
}

#[test]
fn assign_embeddings_matches_python_fixture() {
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let embeddings: Array3<f32> = load_fixture_array3("pipeline_embeddings_data.npy");
    let train_embeddings: Array2<f32> = load_fixture_array2("pipeline_train_embeddings.npy");
    let gamma: Array2<f64> = load_fixture_array2("pipeline_vbx_gamma.npy");
    let pi: Array1<f64> = load_fixture_array1("pipeline_vbx_pi.npy");
    let expected: Array2<i8> = load_fixture_array2("pipeline_hard_clusters.npy");

    let kept_speakers: Vec<usize> = pi
        .iter()
        .enumerate()
        .filter_map(|(idx, weight)| (*weight > 1e-7).then_some(idx))
        .collect();
    let centroids = weighted_centroids(
        &train_embeddings,
        &gamma.mapv(|value| value as f32),
        &kept_speakers,
    );
    let mut hard_clusters = assign_embeddings(&segmentations, &embeddings, &centroids);
    mark_inactive_speakers(&segmentations, &mut hard_clusters);

    assert_eq!(hard_clusters.dim(), expected.dim());
    for (lhs, rhs) in hard_clusters.iter().zip(expected.iter()) {
        assert_eq!(*lhs as i8, *rhs);
    }
}

#[test]
fn extract_embeddings_matches_python_fixture() {
    let harness = PipelineTestHarness::load();
    let Some(seg_model) = harness.cpu_seg_model() else {
        return;
    };
    let Some(mut emb_model) = harness.cpu_emb_model() else {
        return;
    };
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let expected: Array3<f32> = load_fixture_array3("pipeline_embeddings_data.npy");
    let embeddings =
        extract_embeddings(&seg_model, &mut emb_model, harness.audio(), &segmentations).unwrap();

    assert_embedding_tensor_close(&embeddings, &expected, 5e-3);
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_segmentation_matches_python_fixture() {
    let harness = PipelineTestHarness::load();
    let Some(mut seg_model) = harness.coreml_seg_model() else {
        return;
    };
    let expected: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let powerset = PowersetMapping::new(3, 2);
    let raw_windows = seg_model.run(harness.audio()).unwrap();
    let segmentations = decode_windows(raw_windows, &powerset);

    assert_segmentation_tensor_matches(&segmentations, &expected);
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_embeddings_match_python_fixture() {
    let harness = PipelineTestHarness::load();
    let Some(seg_model) = harness.cpu_seg_model() else {
        return;
    };
    let Some(mut emb_model) = harness.coreml_emb_model() else {
        return;
    };
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let expected: Array3<f32> = load_fixture_array3("pipeline_embeddings_data.npy");
    let embeddings =
        extract_embeddings(&seg_model, &mut emb_model, harness.audio(), &segmentations).unwrap();

    assert_embedding_tensor_close(&embeddings, &expected, 5e-3);
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_split_primary_batch_matches_single_tail_path() {
    let harness = PipelineTestHarness::load();
    let Some(seg_model) = harness.cpu_seg_model() else {
        return;
    };
    let Some(mut emb_model) = harness.coreml_emb_model() else {
        return;
    };
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let mut fbanks = Vec::new();
    let mut weights = Vec::new();
    let mut expected = Vec::new();

    'outer: for chunk_idx in 0..segmentations.shape()[0] {
        let chunk_audio = chunk_audio(harness.audio(), &seg_model, chunk_idx);
        let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
        let clean_masks = clean_masks(&chunk_segmentations);
        let fbank = emb_model.compute_chunk_fbank(chunk_audio).unwrap();

        for speaker_idx in 0..chunk_segmentations.ncols() {
            let mask = chunk_segmentations.column(speaker_idx).to_owned();
            let clean_mask = clean_masks.column(speaker_idx).to_owned();
            let used_mask = emb_model
                .select_chunk_mask(
                    mask.as_slice().unwrap(),
                    Some(clean_mask.as_slice().unwrap()),
                    chunk_audio.len(),
                )
                .to_vec();
            expected.push(
                emb_model
                    .embed_masked(
                        chunk_audio,
                        mask.as_slice().unwrap(),
                        Some(clean_mask.as_slice().unwrap()),
                    )
                    .unwrap(),
            );
            fbanks.push(fbank.clone());
            weights.push(used_mask);
            if fbanks.len() == emb_model.split_primary_batch_size() {
                break 'outer;
            }
        }
    }

    assert_eq!(fbanks.len(), emb_model.split_primary_batch_size());
    let batch_inputs: Vec<_> = fbanks
        .iter()
        .zip(weights.iter())
        .map(
            |(fbank, weights)| crate::inference::embedding::SplitTailInput {
                fbank,
                weights: weights.as_slice(),
            },
        )
        .collect();
    let batched = emb_model.embed_tail_batch_inputs(&batch_inputs).unwrap();

    for (row_idx, expected_row) in expected.iter().enumerate() {
        for dim_idx in 0..expected_row.len() {
            let lhs = batched[[row_idx, dim_idx]];
            let rhs = expected_row[dim_idx];
            if (lhs - rhs).abs() > 5e-3 || lhs.is_nan() != rhs.is_nan() {
                panic!("row={row_idx} dim={dim_idx} left={lhs} right={rhs}");
            }
        }
    }
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_single_embedding_matches_python_fixture() {
    let harness = PipelineTestHarness::load();
    let Some(seg_model) = harness.cpu_seg_model() else {
        return;
    };
    let Some(mut emb_model) = harness.coreml_emb_model() else {
        return;
    };
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let expected: Array3<f32> = load_fixture_array3("pipeline_embeddings_data.npy");
    let chunk_idx = 0;
    let speaker_idx = 1;
    let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
    let clean = clean_masks(&chunk_segmentations);
    let mask = chunk_segmentations.column(speaker_idx).to_vec();
    let clean_mask = clean.column(speaker_idx).to_vec();
    let embedding = emb_model
        .embed_masked(
            chunk_audio(harness.audio(), &seg_model, chunk_idx),
            &mask,
            Some(&clean_mask),
        )
        .unwrap();

    for dim_idx in 0..embedding.len() {
        let lhs = embedding[dim_idx];
        let rhs = expected[[chunk_idx, speaker_idx, dim_idx]];
        if (lhs - rhs).abs() > 5e-4 || lhs.is_nan() != rhs.is_nan() {
            panic!("dim={dim_idx} left={lhs} right={rhs}");
        }
    }
}

#[test]
fn run_inference_only_plus_finish_matches_run_with_config() {
    let harness = PipelineTestHarness::load();
    let Some(mut pipeline) = harness.cpu_pipeline() else {
        return;
    };
    let config = pipeline.pipeline_config();
    let combined = pipeline
        .run_with_config(harness.audio(), "file1", &config)
        .unwrap();

    let artifacts = pipeline.run_inference_only(harness.audio()).unwrap();
    let split = pipeline.finish_post_inference(artifacts, &config).unwrap();

    assert_eq!(combined.segments, split.segments);
}

#[test]
fn float_parity_comparison_handles_inactive_embedding_rows_only_as_paired_nan() {
    let matching_nan_actual = Array1::from_vec(vec![0.25, f32::NAN]);
    let matching_nan_expected = Array1::from_vec(vec![0.25 + 1e-6, f32::NAN]);
    assert!(
        compare_f32_arrays(
            "embeddings",
            matching_nan_actual.view().into_dyn(),
            matching_nan_expected.view().into_dyn(),
            1e-5,
            1e-5,
            true,
        )
        .is_ok()
    );

    let unmatched_nan = Array1::from_vec(vec![0.25, f32::NAN]);
    let finite = Array1::from_vec(vec![0.25, 0.0]);
    assert!(
        compare_f32_arrays(
            "embeddings",
            unmatched_nan.view().into_dyn(),
            finite.view().into_dyn(),
            1e-5,
            1e-5,
            true,
        )
        .is_err()
    );

    let infinity = Array1::from_vec(vec![f32::INFINITY]);
    let finite = Array1::from_vec(vec![1.0]);
    assert!(
        compare_f32_arrays(
            "segmentations",
            infinity.view().into_dyn(),
            finite.view().into_dyn(),
            1e-5,
            1e-5,
            false,
        )
        .is_err()
    );

    let too_far = Array1::from_vec(vec![0.26]);
    let reference = Array1::from_vec(vec![0.25]);
    assert!(
        compare_f32_arrays(
            "embeddings",
            too_far.view().into_dyn(),
            reference.view().into_dyn(),
            1e-5,
            1e-5,
            true,
        )
        .is_err()
    );
}

/// Run explicitly with `--ignored` and a fixture overlay containing test.wav and the pinned
/// CPU model files. This deliberately bypasses the suite's skip-on-missing-dynamic-runtime
/// helpers: this test must execute both complete pipelines or fail with a setup/model error.
#[test]
#[ignore = "requires the pinned Community-1 CPU model fixture and statically linked ONNX Runtime"]
fn cpu_slice_and_source_match_complete_pipeline_result() {
    assert!(
        !cfg!(feature = "load-dynamic"),
        "run this parity check without the load-dynamic feature so ORT is statically linked"
    );

    let audio_path = fixture_path("test.wav");
    assert!(
        audio_path.is_file(),
        "missing parity audio fixture: {}",
        audio_path.display()
    );
    let harness = PipelineTestHarness::load();
    let required_cpu_models = [
        "segmentation-3.0.onnx",
        "wespeaker-voxceleb-resnet34.onnx",
        "wespeaker-voxceleb-resnet34.onnx.data",
        "plda_lda.npy",
        "plda_tr.npy",
        "plda_mu.npy",
        "plda_psi.npy",
        "plda_mean1.npy",
        "plda_mean2.npy",
        "wespeaker-voxceleb-resnet34.min_num_samples.txt",
    ];
    for model in required_cpu_models {
        let path = harness.models_dir().join(model);
        assert!(
            path.is_file(),
            "missing pinned CPU model fixture: {}",
            path.display()
        );
    }

    let mut pipeline = PipelineBuilder::from_dir(harness.models_dir(), ExecutionMode::Cpu)
        .build()
        .expect("statically linked ONNX Runtime and pinned CPU model files must load");
    let audio = harness.audio();
    assert!(
        audio.len() > 160_000,
        "fixture should exercise multiple segmentation windows"
    );

    let slice_result = pipeline
        .run_with_file_id(audio, "community1-parity")
        .expect("slice-backed complete CPU pipeline must run");
    let source = crate::audio_source::SliceAudioSource::new(audio);
    let source_result = pipeline
        .run_source(&source, "community1-parity")
        .expect("source-backed complete CPU pipeline must run");

    assert!(
        !slice_result.segments.is_empty(),
        "fixture must produce diarized speech"
    );
    assert_eq!(slice_result.speaker_count.0, source_result.speaker_count.0);
    assert_eq!(slice_result.hard_clusters.0, source_result.hard_clusters.0);
    assert_eq!(slice_result.segments, source_result.segments);
    assert_eq!(
        slice_result.rttm("community1-parity"),
        source_result.rttm("community1-parity")
    );

    assert_f32_arrays_close(
        "segmentations",
        slice_result.segmentations.0.view().into_dyn(),
        source_result.segmentations.0.view().into_dyn(),
        1e-5,
        1e-5,
        false,
    );
    assert_f32_arrays_close(
        "embeddings",
        slice_result.embeddings.0.view().into_dyn(),
        source_result.embeddings.0.view().into_dyn(),
        1e-5,
        1e-5,
        true,
    );
    assert_eq!(
        slice_result.discrete_diarization.0,
        source_result.discrete_diarization.0
    );
}

#[cfg(any(feature = "cuda", feature = "migraphx"))]
struct FailingQualificationSource {
    sample_count: usize,
    failure: AudioSourceError,
    reads: AtomicUsize,
}

#[cfg(any(feature = "cuda", feature = "migraphx"))]
impl AudioSource for FailingQualificationSource {
    fn sample_count(&self) -> usize {
        self.sample_count
    }

    fn read_exact_samples(
        &self,
        _start: usize,
        _destination: &mut [f32],
    ) -> Result<(), AudioSourceError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        Err(self.failure)
    }
}

#[cfg(any(feature = "cuda", feature = "migraphx"))]
fn assert_accelerator_source_results_match(
    mode_name: &str,
    slice_result: &DiarizationResult,
    source_result: &DiarizationResult,
    file_id: &str,
) {
    assert!(
        !slice_result.segments.is_empty(),
        "{mode_name} fixture must produce diarized speech"
    );
    assert_eq!(slice_result.speaker_count.0, source_result.speaker_count.0);
    assert_eq!(slice_result.hard_clusters.0, source_result.hard_clusters.0);
    assert_eq!(slice_result.segments, source_result.segments);
    assert_eq!(slice_result.rttm(file_id), source_result.rttm(file_id));
    assert_f32_arrays_close(
        &format!("{mode_name} segmentations"),
        slice_result.segmentations.0.view().into_dyn(),
        source_result.segmentations.0.view().into_dyn(),
        1e-5,
        1e-5,
        false,
    );
    assert_f32_arrays_close(
        &format!("{mode_name} embeddings"),
        slice_result.embeddings.0.view().into_dyn(),
        source_result.embeddings.0.view().into_dyn(),
        1e-5,
        1e-5,
        true,
    );
    assert_eq!(
        slice_result.discrete_diarization.0,
        source_result.discrete_diarization.0
    );
}

#[cfg(any(feature = "cuda", feature = "migraphx"))]
fn assert_accelerator_source_qualification(mode: ExecutionMode, mode_name: &str) {
    assert!(
        !cfg!(feature = "load-dynamic"),
        "run accelerator source qualification with statically linked ONNX Runtime"
    );

    let audio_path = fixture_path("test.wav");
    assert!(
        audio_path.is_file(),
        "missing parity audio fixture: {}",
        audio_path.display()
    );
    let harness = PipelineTestHarness::load();
    for model in [
        "segmentation-3.0.onnx",
        "segmentation-3.0-b32.onnx",
        "wespeaker-voxceleb-resnet34.onnx",
        "wespeaker-voxceleb-resnet34.onnx.data",
        "wespeaker-fbank.onnx",
        "wespeaker-fbank-b32.onnx",
        "wespeaker-multimask-tail.onnx",
        "wespeaker-multimask-tail-b32.onnx",
        "wespeaker-voxceleb-resnet34-b64.onnx",
        "plda_lda.npy",
        "plda_tr.npy",
        "plda_mu.npy",
        "plda_psi.npy",
        "plda_mean1.npy",
        "plda_mean2.npy",
        "wespeaker-voxceleb-resnet34.min_num_samples.txt",
    ] {
        let path = harness.models_dir().join(model);
        assert!(
            path.is_file(),
            "missing pinned {mode_name} prerequisite: {}",
            path.display()
        );
    }

    let audio = harness.audio();
    assert!(audio.len() > 160_000);
    let mut pipeline = PipelineBuilder::from_dir(harness.models_dir(), mode)
        .build()
        .unwrap_or_else(|error| {
            panic!("pinned {mode_name} provider and model files must load: {error:#}")
        });
    assert_eq!(
        pipeline.seg_model.mode(),
        mode,
        "{mode_name} configured execution mode changed during load"
    );

    let short_slice = pipeline
        .run_with_file_id(audio, "community1-accelerator-short")
        .unwrap_or_else(|error| panic!("slice-backed {mode_name} pipeline failed: {error:#}"));
    let short_source = SliceAudioSource::new(audio);
    let short_range = pipeline
        .run_source(&short_source, "community1-accelerator-short")
        .unwrap_or_else(|error| panic!("source-backed {mode_name} pipeline failed: {error:#}"));
    assert_accelerator_source_results_match(
        mode_name,
        &short_slice,
        &short_range,
        "community1-accelerator-short",
    );
    assert_eq!(
        pipeline.seg_model.mode(),
        mode,
        "{mode_name} configured execution mode changed after parity run"
    );

    // Cross the generic concurrent path's selected embedding batch boundary
    // and leave a padded final segmentation window. This uses the active GPU
    // model's ordinary batch geometry, independently of CoreML chunk capacity.
    let source_batch_size = if pipeline.emb_model.prefers_multi_mask_path()
        && pipeline.emb_model.multi_mask_batch_size() > 0
    {
        pipeline.emb_model.multi_mask_batch_size()
    } else if pipeline.emb_model.prefers_chunk_embedding_path()
        && pipeline.emb_model.split_primary_batch_size() > 0
    {
        pipeline.emb_model.split_primary_batch_size()
    } else {
        pipeline.emb_model.primary_batch_size()
    };
    assert!(source_batch_size > 0, "accelerator model must expose a batch size");
    let boundary_len = pipeline
        .seg_model
        .window_samples()
        .checked_add(
            source_batch_size
                .checked_sub(1)
                .and_then(|windows| windows.checked_mul(pipeline.seg_model.step_samples()))
                .expect("boundary step count"),
        )
        .and_then(|samples| samples.checked_add(1))
        .expect("accelerator boundary fixture length must fit usize");
    let boundary_audio: Vec<f32> = audio
        .iter()
        .copied()
        .cycle()
        .take(boundary_len)
        .collect();
    let boundary_plan = crate::audio_source::AudioWindowPlan::new(
        boundary_audio.len(),
        pipeline.seg_model.window_samples(),
        pipeline.seg_model.step_samples(),
    )
    .expect("accelerator boundary fixture must have a valid window plan");
    assert_eq!(
        boundary_plan.full_window_count(),
        source_batch_size,
        "fixture must contain exactly B complete windows"
    );
    assert!(
        boundary_plan.has_padded_tail(),
        "fixture must include one padded tail window"
    );
    assert_eq!(
        boundary_plan.window_count(),
        source_batch_size + 1,
        "fixture must contain B+1 total windows"
    );
    let boundary_slice = pipeline
        .run_with_file_id(&boundary_audio, "community1-accelerator-capacity-tail")
        .unwrap_or_else(|error| {
            panic!("slice-backed {mode_name} boundary pipeline failed: {error:#}")
        });
    let boundary_source = SliceAudioSource::new(&boundary_audio);
    let boundary_range = pipeline
        .run_source(&boundary_source, "community1-accelerator-capacity-tail")
        .unwrap_or_else(|error| {
            panic!("source-backed {mode_name} boundary pipeline failed: {error:#}")
        });
    assert_accelerator_source_results_match(
        mode_name,
        &boundary_slice,
        &boundary_range,
        "community1-accelerator-capacity-tail",
    );
    assert_eq!(
        pipeline.seg_model.mode(),
        mode,
        "{mode_name} configured execution mode changed after tail parity run"
    );

    for failure in [AudioSourceError::ReadFailed, AudioSourceError::Cancelled] {
        let failing = FailingQualificationSource {
            sample_count: boundary_audio.len(),
            failure,
            reads: AtomicUsize::new(0),
        };
        let error =
            match pipeline.run_source(&failing, "community1-accelerator-source-failure") {
                Ok(_) => panic!("{mode_name} source failure/cancellation unexpectedly succeeded"),
                Err(error) => error,
            };
        assert!(
            error.to_string().contains(&failure.to_string()),
            "{mode_name} source error lost its category ({failure}): {error:#}"
        );
        assert!(
            failing.reads.load(Ordering::Relaxed) == 1,
            "{mode_name} must return the immediate first-read source error without retrying"
        );
        assert_eq!(
            pipeline.seg_model.mode(),
            mode,
            "{mode_name} configured execution mode changed after source error"
        );
    }
}

/// Run on NVIDIA hardware with the pinned GPU model fixture. Verifies complete slice/source
/// parity, exactly B full windows plus one padded tail, and immediate first-read failure/cancel.
/// The mode assertion checks the configured/requested mode value; it does not inspect actual
/// ONNX Runtime execution-provider placement.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires NVIDIA CUDA hardware, pinned accelerator model fixture, and static ONNX Runtime"]
fn cuda_slice_and_source_match_complete_pipeline_and_preserve_configured_mode_on_errors() {
    assert_accelerator_source_qualification(ExecutionMode::Cuda, "CUDA");
}

/// Run on NVIDIA hardware with the pinned GPU model fixture. Verifies the fast
/// CUDA configured mode through the same source/slice checks; this does not inspect
/// actual ONNX Runtime execution-provider placement.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires NVIDIA CUDA hardware, pinned accelerator model fixture, and static ONNX Runtime"]
fn cuda_fast_slice_and_source_match_complete_pipeline_and_preserve_configured_mode_on_errors() {
    assert_accelerator_source_qualification(ExecutionMode::CudaFast, "CUDA fast");
}

/// Run on AMD ROCm/MIGraphX hardware with the pinned GPU model fixture. Verifies complete
/// slice/source parity, exactly B full windows plus one padded tail, and immediate first-read
/// failure/cancel. The mode assertion checks the configured/requested mode value; it does not
/// inspect actual ONNX Runtime execution-provider placement.
#[cfg(feature = "migraphx")]
#[test]
#[ignore = "requires AMD MIGraphX hardware, pinned accelerator model fixture, and static ONNX Runtime"]
fn migraphx_slice_and_source_match_complete_pipeline_and_preserve_configured_mode_on_errors() {
    assert_accelerator_source_qualification(ExecutionMode::MiGraphX, "MiGraphX");
}

/// Run explicitly with `--ignored` using the pinned CoreML model overlay. Both
/// paths use the same native CoreML pipeline, so this checks source-backed
/// window reads against the slice-backed baseline without comparing providers.
#[cfg(feature = "coreml")]
#[test]
#[ignore = "requires the pinned Community-1 CoreML model fixture and statically linked ONNX Runtime"]
fn coreml_slice_and_source_match_complete_pipeline_result() {
    assert!(
        !cfg!(feature = "load-dynamic"),
        "run this parity check without the load-dynamic feature so ORT is statically linked"
    );

    let audio_path = fixture_path("test.wav");
    assert!(
        audio_path.is_file(),
        "missing parity audio fixture: {}",
        audio_path.display()
    );
    let harness = PipelineTestHarness::load();
    for model in [
        "segmentation-3.0.onnx",
        "segmentation-3.0-b32.onnx",
        "wespeaker-voxceleb-resnet34.onnx",
        "wespeaker-voxceleb-resnet34.onnx.data",
        "wespeaker-fbank.onnx",
        "wespeaker-fbank-b32.onnx",
        "wespeaker-voxceleb-resnet34-tail.onnx",
        "wespeaker-voxceleb-resnet34-tail-b3.onnx",
        "wespeaker-voxceleb-resnet34-tail-b32.onnx",
    ] {
        let path = harness.models_dir().join(model);
        assert!(
            path.is_file(),
            "missing pinned CoreML prerequisite: {}",
            path.display()
        );
    }
    for model in [
        "segmentation-3.0.mlmodelc",
        "segmentation-3.0-b32.mlmodelc",
        "segmentation-3.0-b64.mlmodelc",
        "wespeaker-fbank.mlmodelc",
        "wespeaker-fbank-b32.mlmodelc",
        "wespeaker-fbank-30s.mlmodelc",
        "wespeaker-multimask-tail-b32.mlmodelc",
        "wespeaker-voxceleb-resnet34-tail.mlmodelc",
        "wespeaker-voxceleb-resnet34-tail-b3.mlmodelc",
        "wespeaker-voxceleb-resnet34-tail-b32.mlmodelc",
        "wespeaker-chunk-emb-s12-w22.mlmodelc",
        "wespeaker-chunk-emb-s12-w37.mlmodelc",
        "wespeaker-chunk-emb-s12-w53.mlmodelc",
        "wespeaker-chunk-emb-s12-w84.mlmodelc",
        "wespeaker-chunk-emb-s12-w116.mlmodelc",
    ] {
        let path = harness.models_dir().join(model);
        assert!(
            path.is_dir(),
            "missing pinned CoreML model bundle: {}",
            path.display()
        );
    }
    let audio = harness.audio();
    assert!(
        audio.len() > 160_000,
        "fixture should exercise multiple segmentation windows"
    );

    let mut pipeline = PipelineBuilder::from_dir(harness.models_dir(), ExecutionMode::CoreMl)
        .build()
        .expect("native CoreML and pinned CoreML model files must load");
    assert_eq!(pipeline.seg_model.mode(), ExecutionMode::CoreMl);
    let chunk_capacity = pipeline
        .emb_model
        .chunk_window_capacity()
        .expect("pinned CoreML models must expose a chunk embedding capacity");
    let total_windows =
        (audio.len() - pipeline.seg_model.window_samples()) / pipeline.seg_model.step_samples() + 1;
    assert_eq!(total_windows, 17, "fixture targets the short-audio branch");
    assert!(
        total_windows < chunk_capacity,
        "fixture must select sequential chunk embedding (windows={total_windows}, capacity={chunk_capacity})"
    );

    let slice_result = pipeline
        .run_with_file_id(audio, "community1-parity-coreml")
        .expect("slice-backed complete CoreML pipeline must run");
    let source = crate::audio_source::SliceAudioSource::new(audio);
    let source_result = pipeline
        .run_source(&source, "community1-parity-coreml")
        .expect("source-backed complete CoreML pipeline must run");

    assert_coreml_pipeline_results_match(&slice_result, &source_result, "community1-parity-coreml");

    let boundary_sample_count = pipeline
        .seg_model
        .window_samples()
        .checked_add((chunk_capacity - 1) * pipeline.seg_model.step_samples())
        .and_then(|samples| samples.checked_add(1))
        .expect("capacity-boundary fixture length must fit usize");
    let boundary_audio: Vec<f32> = audio
        .iter()
        .copied()
        .cycle()
        .take(boundary_sample_count)
        .collect();
    let boundary_windows = (boundary_audio.len() - pipeline.seg_model.window_samples())
        / pipeline.seg_model.step_samples()
        + 1;
    assert_eq!(boundary_windows, chunk_capacity);
    assert!(boundary_audio.len() > chunk_capacity * pipeline.seg_model.step_samples());

    let boundary_slice_result = pipeline
        .run_with_file_id(&boundary_audio, "community1-capacity-tail-coreml")
        .expect("slice-backed full-capacity CoreML pipeline must run");
    let boundary_source = crate::audio_source::SliceAudioSource::new(&boundary_audio);
    let boundary_source_result = pipeline
        .run_source(&boundary_source, "community1-capacity-tail-coreml")
        .expect("source-backed full-capacity CoreML pipeline must run");
    assert_coreml_pipeline_results_match(
        &boundary_slice_result,
        &boundary_source_result,
        "community1-capacity-tail-coreml",
    );
}

#[test]
fn pipeline_builder_applies_custom_default_config_to_build() {
    let harness = PipelineTestHarness::load();
    let expected = custom_pipeline_config();
    let Some(pipeline) = build_pipeline_or_skip(
        PipelineBuilder::from_dir(harness.models_dir(), ExecutionMode::Cpu)
            .pipeline(expected.clone())
            .build(),
    ) else {
        return;
    };

    let actual = pipeline.pipeline_config();
    assert_eq!(actual.merge_gap, expected.merge_gap);
    assert_eq!(
        actual.speaker_keep_threshold,
        expected.speaker_keep_threshold
    );
    assert_eq!(actual.reconstruct_method, expected.reconstruct_method);
}

#[test]
fn borrowed_pipeline_new_with_config_stores_custom_default_config() {
    let harness = PipelineTestHarness::load();
    let Some(mut seg_model) = harness.cpu_seg_model() else {
        return;
    };
    let Some(mut emb_model) = harness.cpu_emb_model() else {
        return;
    };
    let expected = custom_pipeline_config();
    let pipeline = DiarizationPipeline::new_with_config(
        &mut seg_model,
        &mut emb_model,
        harness.models_dir(),
        expected.clone(),
    )
    .unwrap();

    let actual = pipeline.pipeline_config();
    assert_eq!(actual.merge_gap, expected.merge_gap);
    assert_eq!(
        actual.speaker_keep_threshold,
        expected.speaker_keep_threshold
    );
    assert_eq!(actual.reconstruct_method, expected.reconstruct_method);
}

#[cfg(feature = "coreml")]
#[test]
fn chunk_embedding_pipelined_vs_sequential_baseline() {
    let harness = PipelineTestHarness::load();
    let Some(mut pipeline) = harness.coreml_pipeline() else {
        return;
    };

    // multi-chunk audio (triggers pipelined path)
    // run full pipeline twice: chunk embedding path uses try_chunk_embedding
    // which internally picks pipelined vs sequential based on chunk count
    let result_a = pipeline.run(harness.audio()).unwrap();
    let result_b = pipeline.run(harness.audio()).unwrap();

    // both runs should produce identical RTTM
    assert_eq!(result_a.segments, result_b.segments);

    // also verify run_inference_only + finish_post_inference round-trips
    let config = pipeline.pipeline_config();
    let artifacts = pipeline.run_inference_only(harness.audio()).unwrap();
    let result_split = pipeline.finish_post_inference(artifacts, &config).unwrap();
    assert_eq!(result_a.segments, result_split.segments);
}
