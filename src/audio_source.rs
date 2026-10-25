//! RecapIt addition to speakrs 0.5.0: bounded random-access input primitives
//! for whole-recording diarization. The upstream inference and clustering
//! implementations remain unchanged.
//!
//! A source stores no duration-sized index: callers provide one reusable window buffer and
//! request windows by index. This protocol preserves the pipeline's current global ordering and
//! leaves segmentation, clustering, and reconstruction policy to the existing pipeline.

use std::fmt;

/// A fixed, non-sensitive failure category returned by an audio reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSourceError {
    /// The range, destination size, or source length does not match the request.
    InvalidRange,
    /// The range arithmetic cannot be represented by `usize`.
    Overflow,
    /// The backing source could not provide the requested samples.
    ReadFailed,
    /// The backing operation was cancelled by its owner.
    Cancelled,
}

impl fmt::Display for AudioSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidRange => "audio source range is invalid",
            Self::Overflow => "audio source range overflowed",
            Self::ReadFailed => "audio source read failed",
            Self::Cancelled => "audio source read was cancelled",
        })
    }
}

impl std::error::Error for AudioSourceError {}

/// A whole-recording mono source supporting concurrent, exact range reads.
///
/// Implementations must copy exactly `destination.len()` samples beginning at `start`, or return
/// an error. They must not retain the destination. Concurrent calls may interleave and must not
/// share a mutable seek cursor. `sample_count` is stable for the lifetime of the source.
pub trait AudioSource: Send + Sync {
    /// Return the immutable number of samples in this source.
    fn sample_count(&self) -> usize;

    /// Fill `destination` with exactly the samples at `start..start + destination.len()`.
    fn read_exact_samples(
        &self,
        start: usize,
        destination: &mut [f32],
    ) -> Result<(), AudioSourceError>;
}

/// An allocation-free adapter for an existing in-memory mono slice.
pub struct SliceAudioSource<'a> {
    samples: &'a [f32],
}

impl<'a> SliceAudioSource<'a> {
    /// Borrow a mono sample slice as a random-access source.
    pub fn new(samples: &'a [f32]) -> Self {
        Self { samples }
    }
}

impl AudioSource for SliceAudioSource<'_> {
    fn sample_count(&self) -> usize {
        self.samples.len()
    }

    fn read_exact_samples(
        &self,
        start: usize,
        destination: &mut [f32],
    ) -> Result<(), AudioSourceError> {
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

/// A checked, allocation-free plan for the existing segmentation window chronology.
///
/// Full windows begin at `0, step_samples, 2 * step_samples, ...` while the complete window fits.
/// If the next start is still inside a recording longer than one window, one final window is
/// emitted with its missing suffix zero-padded. This matches the existing speakrs segmentation
/// behavior, including zero windows for inputs shorter than one window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioWindowPlan {
    sample_count: usize,
    window_samples: usize,
    step_samples: usize,
    full_windows: usize,
    has_padded_tail: bool,
}

impl AudioWindowPlan {
    /// Build a plan without storing per-window offsets.
    pub fn new(
        sample_count: usize,
        window_samples: usize,
        step_samples: usize,
    ) -> Result<Self, AudioSourceError> {
        if window_samples == 0 || step_samples == 0 {
            return Err(AudioSourceError::InvalidRange);
        }
        let full_windows = if sample_count < window_samples {
            0
        } else {
            (sample_count - window_samples)
                .checked_div(step_samples)
                .and_then(|count| count.checked_add(1))
                .ok_or(AudioSourceError::Overflow)?
        };
        let next_start = full_windows
            .checked_mul(step_samples)
            .ok_or(AudioSourceError::Overflow)?;
        let has_padded_tail = sample_count > window_samples && next_start < sample_count;
        Ok(Self {
            sample_count,
            window_samples,
            step_samples,
            full_windows,
            has_padded_tail,
        })
    }

    /// Return the source length this plan was built for.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Return the exact caller-owned buffer size required by `read_window_into`.
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// Return the number of complete source-backed windows.
    pub fn full_window_count(&self) -> usize {
        self.full_windows
    }

    /// Return whether one padded final window is required.
    pub fn has_padded_tail(&self) -> bool {
        self.has_padded_tail
    }

    /// Return the number of windows without allocating an offset list.
    pub fn window_count(&self) -> usize {
        self.full_windows + usize::from(self.has_padded_tail)
    }

    /// Fill a caller-owned window buffer with the requested full or padded window.
    pub fn read_window_into<S: AudioSource + ?Sized>(
        &self,
        source: &S,
        index: usize,
        destination: &mut [f32],
    ) -> Result<(), AudioSourceError> {
        if destination.len() != self.window_samples || source.sample_count() != self.sample_count {
            return Err(AudioSourceError::InvalidRange);
        }
        if index < self.full_windows {
            let start = index
                .checked_mul(self.step_samples)
                .ok_or(AudioSourceError::Overflow)?;
            let end = start
                .checked_add(self.window_samples)
                .ok_or(AudioSourceError::Overflow)?;
            if end > self.sample_count {
                return Err(AudioSourceError::InvalidRange);
            }
            return source.read_exact_samples(start, destination);
        }
        if index != self.full_windows || !self.has_padded_tail {
            return Err(AudioSourceError::InvalidRange);
        }
        let start = self
            .full_windows
            .checked_mul(self.step_samples)
            .ok_or(AudioSourceError::Overflow)?;
        let available = self
            .sample_count
            .checked_sub(start)
            .filter(|count| *count > 0 && *count < self.window_samples)
            .ok_or(AudioSourceError::InvalidRange)?;
        destination.fill(0.0);
        source.read_exact_samples(start, &mut destination[..available])
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioSource, AudioSourceError, AudioWindowPlan, SliceAudioSource};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    fn reference_windows(audio: &[f32], window: usize, step: usize) -> Vec<Vec<f32>> {
        let mut offsets = Vec::new();
        let mut offset = 0usize;
        while offset + window <= audio.len() {
            offsets.push(offset);
            offset += step;
        }
        let mut expected: Vec<Vec<f32>> = offsets
            .into_iter()
            .map(|start| audio[start..start + window].to_vec())
            .collect();
        if offset < audio.len() && audio.len() > window {
            let mut padded = vec![0.0; window];
            padded[..audio.len() - offset].copy_from_slice(&audio[offset..]);
            expected.push(padded);
        }
        expected
    }

    #[test]
    fn slice_reader_matches_legacy_window_order_and_padding() {
        for window in 1..8 {
            for step in 1..10 {
                for length in 0..40 {
                    let audio: Vec<f32> = (0..length).map(|index| index as f32 + 0.25).collect();
                    let source = SliceAudioSource::new(&audio);
                    let plan = AudioWindowPlan::new(length, window, step).unwrap();
                    let expected = reference_windows(&audio, window, step);
                    assert_eq!(
                        plan.window_count(),
                        expected.len(),
                        "len={length} window={window} step={step}"
                    );
                    let mut actual = vec![f32::NAN; window];
                    for (index, expected_window) in expected.iter().enumerate() {
                        plan.read_window_into(&source, index, &mut actual).unwrap();
                        assert_eq!(
                            &actual, expected_window,
                            "len={length} window={window} step={step} index={index}"
                        );
                    }
                    assert_eq!(
                        plan.read_window_into(&source, expected.len(), &mut actual),
                        Err(AudioSourceError::InvalidRange)
                    );
                }
            }
        }
    }

    struct FormulaSource {
        samples: usize,
        reads: AtomicUsize,
    }

    impl AudioSource for FormulaSource {
        fn sample_count(&self) -> usize {
            self.samples
        }
        fn read_exact_samples(
            &self,
            start: usize,
            destination: &mut [f32],
        ) -> Result<(), AudioSourceError> {
            let end = start
                .checked_add(destination.len())
                .ok_or(AudioSourceError::Overflow)?;
            if end > self.samples {
                return Err(AudioSourceError::InvalidRange);
            }
            for (index, value) in destination.iter_mut().enumerate() {
                *value = (start + index) as f32;
            }
            self.reads.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn huge_recording_plan_has_constant_metadata_and_reads_distant_windows() {
        let count = 10_000_000_000usize;
        let source = FormulaSource {
            samples: count,
            reads: AtomicUsize::new(0),
        };
        let plan = AudioWindowPlan::new(count, 160, 80).unwrap();
        assert_eq!(plan.full_window_count(), 124_999_999);
        assert!(plan.has_padded_tail());
        assert_eq!(plan.window_count(), 125_000_000);
        let mut buffer = [0.0; 160];
        plan.read_window_into(&source, 120_000_000, &mut buffer)
            .unwrap();
        assert_eq!(buffer[0], 9_600_000_000.0);
        assert_eq!(source.reads.load(Ordering::Relaxed), 1);
        assert_eq!(
            std::mem::size_of_val(&plan),
            std::mem::size_of::<AudioWindowPlan>()
        );
    }

    #[test]
    fn checked_configuration_ranges_and_source_size_are_enforced() {
        assert_eq!(
            AudioWindowPlan::new(10, 0, 1),
            Err(AudioSourceError::InvalidRange)
        );
        assert_eq!(
            AudioWindowPlan::new(10, 1, 0),
            Err(AudioSourceError::InvalidRange)
        );
        let plan = AudioWindowPlan::new(10, 4, 3).unwrap();
        let source = SliceAudioSource::new(&[0.0; 9]);
        let mut buffer = [0.0; 4];
        assert_eq!(
            plan.read_window_into(&source, 0, &mut buffer),
            Err(AudioSourceError::InvalidRange)
        );
        let exact = SliceAudioSource::new(&[0.0; 10]);
        assert_eq!(plan.read_window_into(&exact, 0, &mut buffer), Ok(()));
        assert_eq!(
            plan.read_window_into(&exact, 0, &mut [0.0; 3]),
            Err(AudioSourceError::InvalidRange)
        );
        assert_eq!(
            AudioWindowPlan::new(usize::MAX, 1, usize::MAX - 1),
            Err(AudioSourceError::Overflow)
        );
    }

    struct FailingSource;
    impl AudioSource for FailingSource {
        fn sample_count(&self) -> usize {
            8
        }
        fn read_exact_samples(&self, _: usize, _: &mut [f32]) -> Result<(), AudioSourceError> {
            Err(AudioSourceError::Cancelled)
        }
    }

    #[test]
    fn source_failure_and_cancellation_are_propagated_without_fallback() {
        let plan = AudioWindowPlan::new(8, 4, 2).unwrap();
        assert_eq!(
            plan.read_window_into(&FailingSource, 0, &mut [0.0; 4]),
            Err(AudioSourceError::Cancelled)
        );
    }

    struct BarrierSource {
        samples: usize,
        active: AtomicUsize,
        peak: AtomicUsize,
        all_readers_entered: Arc<Barrier>,
    }

    impl AudioSource for BarrierSource {
        fn sample_count(&self) -> usize {
            self.samples
        }

        fn read_exact_samples(
            &self,
            start: usize,
            destination: &mut [f32],
        ) -> Result<(), AudioSourceError> {
            let end = start
                .checked_add(destination.len())
                .ok_or(AudioSourceError::Overflow)?;
            if end > self.samples {
                return Err(AudioSourceError::InvalidRange);
            }
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(active, Ordering::SeqCst);
            self.all_readers_entered.wait();
            for (offset, value) in destination.iter_mut().enumerate() {
                *value = (start + offset) as f32;
            }
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn four_concurrent_reader_calls_are_active_before_any_can_finish() {
        const READERS: usize = 4;
        let source = Arc::new(BarrierSource {
            samples: READERS * 4,
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            all_readers_entered: Arc::new(Barrier::new(READERS)),
        });
        let plan = AudioWindowPlan::new(READERS * 4, 4, 4).unwrap();
        let mut workers = Vec::with_capacity(READERS);
        for index in 0..READERS {
            let source = Arc::clone(&source);
            workers.push(std::thread::spawn(move || {
                let mut window = [f32::NAN; 4];
                plan.read_window_into(source.as_ref(), index, &mut window)
                    .unwrap();
                window
            }));
        }
        for (index, worker) in workers.into_iter().enumerate() {
            assert_eq!(
                worker.join().unwrap(),
                [
                    (index * 4) as f32,
                    (index * 4 + 1) as f32,
                    (index * 4 + 2) as f32,
                    (index * 4 + 3) as f32,
                ]
            );
        }
        assert_eq!(source.peak.load(Ordering::SeqCst), READERS);
        assert_eq!(source.active.load(Ordering::SeqCst), 0);
    }

    struct CancelWhileBlockedSource {
        cancelled: Arc<AtomicBool>,
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
    }

    impl AudioSource for CancelWhileBlockedSource {
        fn sample_count(&self) -> usize {
            4
        }

        fn read_exact_samples(
            &self,
            start: usize,
            destination: &mut [f32],
        ) -> Result<(), AudioSourceError> {
            if start
                .checked_add(destination.len())
                .filter(|end| *end <= 4)
                .is_none()
            {
                return Err(AudioSourceError::InvalidRange);
            }
            self.entered.wait();
            self.release.wait();
            if self.cancelled.load(Ordering::SeqCst) {
                return Err(AudioSourceError::Cancelled);
            }
            destination.fill(1.0);
            Ok(())
        }
    }

    #[test]
    fn cancellation_during_blocked_read_propagates_without_partial_window() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let source = Arc::new(CancelWhileBlockedSource {
            cancelled: Arc::clone(&cancelled),
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        });
        let plan = AudioWindowPlan::new(4, 4, 4).unwrap();
        let worker = std::thread::spawn(move || {
            let mut window = [f32::NAN; 4];
            let result = plan.read_window_into(source.as_ref(), 0, &mut window);
            (result, window)
        });
        entered.wait();
        cancelled.store(true, Ordering::SeqCst);
        release.wait();
        let (result, window) = worker.join().unwrap();
        assert_eq!(result, Err(AudioSourceError::Cancelled));
        assert!(window.iter().all(|sample| sample.is_nan()));
    }
}
