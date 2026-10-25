#[cfg(feature = "coreml")]
use ndarray::{Array2, Array3};

use super::SegmentationError;
use crate::audio_source::{AudioSource, AudioWindowPlan};

pub(super) type OutputShape3 = (usize, usize, usize);

pub(super) struct SegmentationWindows<'a> {
    audio: &'a [f32],
    offsets: Vec<usize>,
    padded: Option<Vec<f32>>,
    window_samples: usize,
}

/// Window access shared by slice and random-access sources. Implementations fill the bounded
/// destination directly so parallel CoreML workers never need a recording-sized raw-window list.
pub(super) trait SegmentationWindowReader: Sync {
    fn total_windows(&self) -> usize;
    fn read_window_into(
        &self,
        idx: usize,
        destination: &mut [f32],
        context: &'static str,
    ) -> Result<(), SegmentationError>;
}

pub(super) struct SourceSegmentationWindows<'a, S: AudioSource + ?Sized> {
    source: &'a S,
    plan: AudioWindowPlan,
}

impl<'a, S: AudioSource + ?Sized> SourceSegmentationWindows<'a, S> {
    pub(super) fn new(
        source: &'a S,
        window_samples: usize,
        step_samples: usize,
    ) -> Result<Self, SegmentationError> {
        Ok(Self {
            source,
            plan: AudioWindowPlan::new(source.sample_count(), window_samples, step_samples)?,
        })
    }
}

impl<S: AudioSource + ?Sized> SegmentationWindowReader for SourceSegmentationWindows<'_, S> {
    fn total_windows(&self) -> usize {
        self.plan.window_count()
    }

    fn read_window_into(
        &self,
        idx: usize,
        destination: &mut [f32],
        _context: &'static str,
    ) -> Result<(), SegmentationError> {
        self.plan.read_window_into(self.source, idx, destination)?;
        Ok(())
    }
}

impl<'a> SegmentationWindows<'a> {
    pub(super) fn collect(audio: &'a [f32], window_samples: usize, step_samples: usize) -> Self {
        let mut offsets = Vec::new();
        let mut offset = 0;
        while offset + window_samples <= audio.len() {
            offsets.push(offset);
            offset += step_samples;
        }

        let padded = if offset < audio.len() && audio.len() > window_samples {
            let mut padded = vec![0.0f32; window_samples];
            let remaining = audio.len() - offset;
            padded[..remaining].copy_from_slice(&audio[offset..]);
            Some(padded)
        } else {
            None
        };

        Self {
            audio,
            offsets,
            padded,
            window_samples,
        }
    }

    pub(super) fn total_windows(&self) -> usize {
        self.offsets.len() + self.padded.is_some() as usize
    }

    pub(super) fn is_empty(&self) -> bool {
        self.total_windows() == 0
    }

    pub(super) fn window(
        &self,
        idx: usize,
        context: &'static str,
    ) -> Result<&[f32], SegmentationError> {
        if idx < self.offsets.len() {
            let start = self.offsets[idx];
            return Ok(&self.audio[start..start + self.window_samples]);
        }
        if idx == self.offsets.len() {
            return padded_window(&self.padded, context);
        }

        Err(SegmentationError::Invariant {
            context,
            message: format!(
                "window index {idx} exceeded total window count {}",
                self.total_windows()
            ),
        })
    }
}

#[cfg(feature = "coreml")]
impl SegmentationWindowReader for SegmentationWindows<'_> {
    fn total_windows(&self) -> usize {
        self.total_windows()
    }

    fn read_window_into(
        &self,
        idx: usize,
        destination: &mut [f32],
        context: &'static str,
    ) -> Result<(), SegmentationError> {
        let window = self.window(idx, context)?;
        if destination.len() != self.window_samples || window.len() > destination.len() {
            return Err(SegmentationError::Invariant {
                context,
                message: "window destination has an unexpected size".to_owned(),
            });
        }
        destination.fill(0.0);
        destination[..window.len()].copy_from_slice(window);
        Ok(())
    }
}

#[cfg(feature = "coreml")]
pub(super) fn array3_slice<'a>(
    buffer: &'a Array3<f32>,
    context: &'static str,
) -> Result<&'a [f32], SegmentationError> {
    buffer
        .as_slice()
        .ok_or_else(|| SegmentationError::Invariant {
            context,
            message: "input buffer was not contiguous".to_owned(),
        })
}

pub(super) fn padded_window<'a>(
    padded: &'a Option<Vec<f32>>,
    context: &'static str,
) -> Result<&'a [f32], SegmentationError> {
    padded
        .as_deref()
        .ok_or_else(|| SegmentationError::Invariant {
            context,
            message: "missing padded window".to_owned(),
        })
}

pub(super) fn first_output<T>(
    outputs: impl IntoIterator<Item = T>,
    context: &'static str,
) -> Result<T, SegmentationError> {
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| SegmentationError::MalformedOutput {
            context,
            message: "missing output tensor".to_owned(),
        })
}

pub(super) fn output_shape3(
    shape: &ort::value::Shape,
    context: &'static str,
) -> Result<OutputShape3, SegmentationError> {
    let [batch, frames, classes]: [i64; 3] =
        shape
            .as_ref()
            .try_into()
            .map_err(|_| SegmentationError::MalformedOutput {
                context,
                message: format!("expected rank 3 output, got shape {shape}"),
            })?;

    let dims = [batch, frames, classes];
    if dims.iter().any(|dim| *dim < 0) {
        return Err(SegmentationError::MalformedOutput {
            context,
            message: format!("expected non-negative output dimensions, got shape {shape}"),
        });
    }

    Ok((batch as usize, frames as usize, classes as usize))
}

#[cfg(feature = "coreml")]
pub(super) fn segmentation_array(
    frames: usize,
    classes: usize,
    data: Vec<f32>,
    context: &'static str,
) -> Result<Array2<f32>, SegmentationError> {
    Array2::from_shape_vec((frames, classes), data).map_err(|error| SegmentationError::Invariant {
        context,
        message: format!("invalid segmentation output shape: {error}"),
    })
}

#[cfg(feature = "coreml")]
pub(super) fn segmentation_array_from_slice(
    frames: usize,
    classes: usize,
    data: &[f32],
    context: &'static str,
) -> Result<Array2<f32>, SegmentationError> {
    segmentation_array(frames, classes, data.to_vec(), context)
}

#[cfg(feature = "coreml")]
pub(super) fn worker_panic(worker: &'static str) -> SegmentationError {
    SegmentationError::WorkerPanic {
        worker: worker.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{SegmentationWindowReader, SourceSegmentationWindows, first_output, output_shape3};
    use crate::audio_source::{AudioSource, AudioSourceError, SliceAudioSource};

    struct FailingSource {
        samples: Vec<f32>,
        failure: AudioSourceError,
    }

    impl AudioSource for FailingSource {
        fn sample_count(&self) -> usize {
            self.samples.len()
        }

        fn read_exact_samples(
            &self,
            start: usize,
            destination: &mut [f32],
        ) -> Result<(), AudioSourceError> {
            if start >= 4 {
                return Err(self.failure);
            }
            let end = start + destination.len();
            destination.copy_from_slice(
                self.samples
                    .get(start..end)
                    .ok_or(AudioSourceError::InvalidRange)?,
            );
            Ok(())
        }
    }

    #[test]
    fn source_window_reader_matches_slice_order_and_tail_padding() {
        let samples: Vec<f32> = (0..23).map(|sample| sample as f32).collect();
        let slice = SliceAudioSource::new(&samples);
        let windows = SourceSegmentationWindows::new(&slice, 8, 5).unwrap();
        assert_eq!(windows.total_windows(), 5);

        let expected = [
            vec![0., 1., 2., 3., 4., 5., 6., 7.],
            vec![5., 6., 7., 8., 9., 10., 11., 12.],
            vec![10., 11., 12., 13., 14., 15., 16., 17.],
            vec![15., 16., 17., 18., 19., 20., 21., 22.],
            vec![20., 21., 22., 0., 0., 0., 0., 0.],
        ];
        for (index, expected_window) in expected.iter().enumerate() {
            let mut actual = vec![f32::NAN; 8];
            windows
                .read_window_into(index, &mut actual, "test source windows")
                .unwrap();
            assert_eq!(&actual, expected_window);
        }
    }

    #[test]
    fn source_window_reader_propagates_reader_failure_categories() {
        for failure in [AudioSourceError::ReadFailed, AudioSourceError::Cancelled] {
            let source = FailingSource {
                samples: (0..20).map(|sample| sample as f32).collect(),
                failure,
            };
            let windows = SourceSegmentationWindows::new(&source, 8, 4).unwrap();
            let mut buffer = vec![0.0; 8];
            assert!(matches!(
                windows.read_window_into(1, &mut buffer, "test source windows"),
                Err(crate::inference::segmentation::SegmentationError::AudioSource(error))
                    if error == failure
            ));
        }
    }

    #[test]
    fn first_output_reports_missing_tensor() {
        let error = first_output(Vec::<()>::new(), "segmentation test").unwrap_err();

        assert_eq!(
            error.to_string(),
            "segmentation test: missing output tensor"
        );
    }

    #[test]
    fn output_shape3_reports_low_rank_tensor() {
        let shape = ort::value::Shape::from([10_i64, 3]);
        let error = output_shape3(&shape, "segmentation test").unwrap_err();

        assert_eq!(
            error.to_string(),
            "segmentation test: expected rank 3 output, got shape [10, 3]"
        );
    }
}
