use crossbeam_channel::Sender;
use ndarray::Array2;
use ort::value::TensorRef;
use tracing::debug;

use super::{PRIMARY_BATCH_SIZE, SegmentationError, SegmentationModel};
use crate::audio_source::{AudioSource, AudioWindowPlan, SliceAudioSource};
use crate::inference::segmentation::tensor::{SegmentationWindows, first_output, output_shape3};

fn read_window_batch<S: AudioSource + ?Sized>(
    source: &S,
    plan: &AudioWindowPlan,
    first_index: usize,
    window_count: usize,
    batch_capacity: usize,
    window_samples: usize,
) -> Result<Vec<f32>, SegmentationError> {
    if window_count == 0 || window_count > batch_capacity || plan.window_samples() != window_samples
    {
        return Err(SegmentationError::Invariant {
            context: "source-backed segmentation batch",
            message: "window batch bounds do not match the plan".to_owned(),
        });
    }
    let sample_capacity =
        batch_capacity
            .checked_mul(window_samples)
            .ok_or_else(|| SegmentationError::Invariant {
                context: "source-backed segmentation batch",
                message: "window batch size overflowed".to_owned(),
            })?;
    let mut storage = vec![0.0f32; sample_capacity];
    for batch_index in 0..window_count {
        let start = batch_index * window_samples;
        let end = start + window_samples;
        plan.read_window_into(source, first_index + batch_index, &mut storage[start..end])?;
    }
    Ok(storage)
}

impl SegmentationModel {
    /// Run segmentation on audio, streaming raw logits through a channel
    ///
    /// Same logic as `run()`, but sends each decoded window through `tx` as it's produced
    /// instead of collecting into a Vec. Returns total window count
    pub fn run_streaming(
        &mut self,
        audio: &[f32],
        tx: Sender<Array2<f32>>,
    ) -> Result<usize, SegmentationError> {
        self.run_streaming_source(&SliceAudioSource::new(audio), tx)
    }

    /// Run segmentation from a random-access whole-recording source, retaining only the current
    /// bounded batch of windows instead of storing an offset list for the whole recording.
    ///
    /// The source must provide stable sample count and exact concurrent reads as documented by
    /// [`AudioSource`]. A batch session uses at most `PRIMARY_BATCH_SIZE` model windows of scratch.
    pub fn run_streaming_source<S: AudioSource + ?Sized>(
        &mut self,
        source: &S,
        tx: Sender<Array2<f32>>,
    ) -> Result<usize, SegmentationError> {
        let plan = AudioWindowPlan::new(
            source.sample_count(),
            self.window_samples,
            self.step_samples,
        )?;
        let total_windows = plan.window_count();
        if total_windows == 0 {
            return Ok(0);
        }

        let seg_start = std::time::Instant::now();
        let mut seg_infer_time = std::time::Duration::ZERO;
        let mut seg_batched = 0u32;
        let mut seg_single = 0u32;

        let has_batched = self.primary_batched_session.is_some();
        let mut next_idx = 0;
        while next_idx < total_windows {
            let remaining = total_windows - next_idx;

            if remaining >= PRIMARY_BATCH_SIZE && has_batched {
                let batch_storage = read_window_batch(
                    source,
                    &plan,
                    next_idx,
                    PRIMARY_BATCH_SIZE,
                    PRIMARY_BATCH_SIZE,
                    self.window_samples,
                )?;
                let batch: Vec<&[f32]> = batch_storage.chunks_exact(self.window_samples).collect();

                let t = std::time::Instant::now();
                let results = self.run_batch(&batch)?;
                seg_infer_time += t.elapsed();
                seg_batched += 1;
                for r in results {
                    tx.send(r)?;
                }
                next_idx += PRIMARY_BATCH_SIZE;
                continue;
            }

            if remaining > 1 && has_batched {
                let batch_storage = read_window_batch(
                    source,
                    &plan,
                    next_idx,
                    remaining,
                    PRIMARY_BATCH_SIZE,
                    self.window_samples,
                )?;
                let batch: Vec<&[f32]> = batch_storage.chunks_exact(self.window_samples).collect();

                let t = std::time::Instant::now();
                let results = self.run_batch(&batch)?;
                seg_infer_time += t.elapsed();
                seg_batched += 1;
                for r in results.into_iter().take(remaining) {
                    tx.send(r)?;
                }
                next_idx = total_windows;
                continue;
            }

            let mut window = vec![0.0f32; self.window_samples];
            plan.read_window_into(source, next_idx, &mut window)?;
            let t = std::time::Instant::now();
            let result = self.run_window(&window)?;
            seg_infer_time += t.elapsed();
            seg_single += 1;
            tx.send(result)?;
            next_idx += 1;
        }

        let total_seg = seg_start.elapsed();
        debug!(
            windows = total_windows,
            seg_batched,
            seg_single,
            seg_infer_ms = seg_infer_time.as_millis(),
            seg_total_ms = total_seg.as_millis(),
            seg_overhead_ms = (total_seg - seg_infer_time).as_millis(),
            "Segmentation thread profile"
        );

        Ok(total_windows)
    }

    /// Run segmentation on audio, returning raw logits per window
    ///
    /// Returns `Vec<Array2<f32>>` where each element is [frames, 7] logits
    pub fn run(&mut self, audio: &[f32]) -> Result<Vec<Array2<f32>>, ort::Error> {
        let windows = SegmentationWindows::collect(audio, self.window_samples, self.step_samples);
        let total_windows = windows.total_windows();
        let mut results = Vec::with_capacity(total_windows);
        let mut next_idx = 0;

        while next_idx < total_windows {
            let remaining = total_windows - next_idx;
            if remaining >= PRIMARY_BATCH_SIZE && self.primary_batched_session.is_some() {
                let batch: Vec<&[f32]> = (next_idx..next_idx + PRIMARY_BATCH_SIZE)
                    .map(|idx| windows.window(idx, "segmentation run batch window"))
                    .collect::<Result<_, _>>()
                    .map_err(|error| ort::Error::new(error.to_string()))?;
                results.extend(
                    self.run_batch(&batch)
                        .map_err(|error| ort::Error::new(error.to_string()))?,
                );
                next_idx += PRIMARY_BATCH_SIZE;
                continue;
            }

            let window = windows
                .window(next_idx, "segmentation run tail window")
                .map_err(|error| ort::Error::new(error.to_string()))?;
            results.push(
                self.run_window(window)
                    .map_err(|error| ort::Error::new(error.to_string()))?,
            );
            next_idx += 1;
        }

        Ok(results)
    }

    fn run_window(&mut self, window: &[f32]) -> Result<Array2<f32>, SegmentationError> {
        #[cfg(feature = "coreml")]
        if let Some(ref native) = self.native_session {
            return Self::run_native_single(
                native,
                window,
                &mut self.input_buffer,
                &self.cached_single_input_shape,
            )
            .map_err(SegmentationError::Ort);
        }

        self.input_buffer.fill(0.0);
        self.input_buffer
            .slice_mut(ndarray::s![0, 0, ..window.len()])
            .assign(&ndarray::ArrayView1::from(window));
        let input_tensor = TensorRef::from_array_view(self.input_buffer.view())?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;
        let output = first_output(outputs.values(), "segmentation window output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;

        let (_batch, frames, classes) = output_shape3(shape, "segmentation window output")?;

        Array2::from_shape_vec((frames, classes), data.to_vec()).map_err(|error| {
            SegmentationError::MalformedOutput {
                context: "segmentation window output",
                message: format!("invalid output shape: {error}"),
            }
        })
    }

    fn run_batch(&mut self, windows: &[&[f32]]) -> Result<Vec<Array2<f32>>, SegmentationError> {
        #[cfg(feature = "coreml")]
        if let Some(ref native) = self.native_batched_session {
            return Self::run_native_batch(
                native,
                windows,
                &mut self.primary_batch_input_buffer,
                &self.cached_batch_input_shape,
            )
            .map_err(SegmentationError::Ort);
        }

        self.primary_batch_input_buffer.fill(0.0);
        for (batch_idx, window) in windows.iter().enumerate() {
            self.primary_batch_input_buffer
                .slice_mut(ndarray::s![batch_idx, 0, ..window.len()])
                .assign(&ndarray::ArrayView1::from(*window));
        }
        let input_tensor = TensorRef::from_array_view(self.primary_batch_input_buffer.view())?;

        let outputs = self
            .primary_batched_session
            .as_mut()
            .ok_or_else(|| ort::Error::new("missing primary batched segmentation session"))?
            .run(ort::inputs![input_tensor])?;
        let output = first_output(outputs.values(), "segmentation batch output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;

        let (batch, frames, classes) = output_shape3(shape, "segmentation batch output")?;
        let stride = frames * classes;
        let expected_len =
            batch
                .checked_mul(stride)
                .ok_or_else(|| SegmentationError::MalformedOutput {
                    context: "segmentation batch output",
                    message: format!("output shape {shape} exceeded addressable memory"),
                })?;
        if data.len() != expected_len {
            return Err(SegmentationError::MalformedOutput {
                context: "segmentation batch output",
                message: format!(
                    "shape {shape} expected {expected_len} values, got {}",
                    data.len()
                ),
            });
        }

        (0..batch)
            .map(|batch_idx| {
                let start = batch_idx * stride;
                Array2::from_shape_vec((frames, classes), data[start..start + stride].to_vec())
                    .map_err(|error| SegmentationError::MalformedOutput {
                        context: "segmentation batch output",
                        message: format!("invalid output shape: {error}"),
                    })
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

#[cfg(test)]
mod source_batch_tests {
    use super::read_window_batch;
    use crate::audio_source::{AudioSource, AudioSourceError, AudioWindowPlan};
    use crate::inference::segmentation::tensor::SegmentationWindows;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingSource {
        samples: Vec<f32>,
        largest_read: AtomicUsize,
    }

    impl AudioSource for CountingSource {
        fn sample_count(&self) -> usize {
            self.samples.len()
        }

        fn read_exact_samples(
            &self,
            start: usize,
            destination: &mut [f32],
        ) -> Result<(), AudioSourceError> {
            self.largest_read
                .fetch_max(destination.len(), Ordering::Relaxed);
            let end = start
                .checked_add(destination.len())
                .ok_or(AudioSourceError::Overflow)?;
            let source = self
                .samples
                .get(start..end)
                .ok_or(AudioSourceError::InvalidRange)?;
            destination.copy_from_slice(source);
            Ok(())
        }
    }

    #[test]
    fn source_batch_matches_slice_windows_and_bounds_reads_to_window_size() {
        let source = CountingSource {
            samples: (0..23).map(|sample| sample as f32).collect(),
            largest_read: AtomicUsize::new(0),
        };
        let plan = AudioWindowPlan::new(source.sample_count(), 8, 6).unwrap();
        let first = read_window_batch(&source, &plan, 0, 2, 4, 8).unwrap();
        let tail = read_window_batch(&source, &plan, 2, 2, 4, 8).unwrap();
        assert_eq!(
            &first[..16],
            &[
                0., 1., 2., 3., 4., 5., 6., 7., 6., 7., 8., 9., 10., 11., 12., 13.
            ]
        );
        assert_eq!(
            &tail[..16],
            &[
                12., 13., 14., 15., 16., 17., 18., 19., 18., 19., 20., 21., 22., 0., 0., 0.
            ]
        );
        assert!(tail[16..].iter().all(|sample| *sample == 0.0));
        assert_eq!(source.largest_read.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn source_batch_refuses_counts_outside_fixed_capacity() {
        let source = CountingSource {
            samples: vec![1.0; 32],
            largest_read: AtomicUsize::new(0),
        };
        let plan = AudioWindowPlan::new(32, 8, 8).unwrap();
        assert!(read_window_batch(&source, &plan, 0, 0, 4, 8).is_err());
        assert!(read_window_batch(&source, &plan, 0, 5, 4, 8).is_err());
        assert_eq!(source.largest_read.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn source_batches_match_upstream_segmentation_windows_for_lengths_steps_and_tails() {
        for sample_count in 0..64 {
            for window_samples in 1..9 {
                for step_samples in 1..11 {
                    let source = CountingSource {
                        samples: (0..sample_count).map(|i| i as f32 + 0.5).collect(),
                        largest_read: AtomicUsize::new(0),
                    };
                    let plan =
                        AudioWindowPlan::new(sample_count, window_samples, step_samples).unwrap();
                    let upstream =
                        SegmentationWindows::collect(&source.samples, window_samples, step_samples);
                    let mut actual = Vec::new();
                    let mut first = 0;
                    while first < plan.window_count() {
                        let count = (plan.window_count() - first).min(4);
                        let storage =
                            read_window_batch(&source, &plan, first, count, 4, window_samples)
                                .unwrap();
                        actual.extend(
                            storage
                                .chunks_exact(window_samples)
                                .take(count)
                                .map(|window| window.to_vec()),
                        );
                        first += count;
                    }
                    let expected: Vec<Vec<f32>> = (0..upstream.total_windows())
                        .map(|index| {
                            upstream
                                .window(index, "source batch parity test")
                                .unwrap()
                                .to_vec()
                        })
                        .collect();
                    assert_eq!(
                        actual,
                        expected,
                        "samples={sample_count} window={window_samples} step={step_samples}"
                    );
                    assert!(source.largest_read.load(Ordering::Relaxed) <= window_samples);
                }
            }
        }
    }

    struct CancelledSource;

    impl AudioSource for CancelledSource {
        fn sample_count(&self) -> usize {
            32
        }

        fn read_exact_samples(
            &self,
            _start: usize,
            _destination: &mut [f32],
        ) -> Result<(), AudioSourceError> {
            Err(AudioSourceError::Cancelled)
        }
    }

    #[test]
    fn source_batch_propagates_cancellation_without_filling_later_windows() {
        let source = CancelledSource;
        let plan = AudioWindowPlan::new(source.sample_count(), 8, 4).unwrap();
        let error = read_window_batch(&source, &plan, 0, 2, 4, 8).unwrap_err();
        assert!(matches!(
            error,
            super::SegmentationError::AudioSource(AudioSourceError::Cancelled)
        ));
    }
}
