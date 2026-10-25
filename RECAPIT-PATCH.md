# RecapIt patch provenance

This public fork is based on `avencera/speakrs` source commit `96f3cec0756bc9df7f830a9ed31e782ddd445edd`, the source revision recorded in the `speakrs` 0.5.0 crates.io archive's `.cargo_vcs_info.json`. The verified archive SHA-256 is `b7ed3d9e4670beaa31736bc2baf3736aca4554920edeb6ed463c903da9df57ac`; its `Cargo.toml.orig` matches the manifest at that source commit. The archive's generated normalized `Cargo.toml` is packaging metadata and is not used as the fork's source manifest. The upstream `LICENSE` remains unchanged.

The fork disables `ndarray-npy` default features to match the published 0.5.0 package normalization and preserve the app's locked dependency graph; speakrs uses `.npy` files only.

## B11 bounded source-backed processing

The RecapIt delta adds `src/audio_source.rs` and exports it from `src/lib.rs`. `AudioSource` and `AudioWindowPlan` provide exact random-access reads and constant-size window metadata. Segmentation adapters in `src/inference/segmentation.rs` and its `run`, `tensor`, `parallel`, `parallel/batch`, and `parallel/single` modules read the ordered full windows and padded tail without retaining recording-sized offsets or padded audio.

Pipeline integration in `src/pipeline.rs`, `src/pipeline/concurrent.rs`, `src/pipeline/chunk_embedding.rs` and its `gpu`, `orchestrate`, and `prep` modules, plus `src/pipeline/types.rs` and its `data` and `extract` modules, bounds source-backed audio and embedding intermediates by existing inference batch sizes. Masked-embedding scheduling remains capped at 64 rows, split-tail rows batch across chunks at the same 64-row bound, and multi-mask fbank batches retain the 32-chunk schedule. Complete decoded segmentations and embeddings remain available to the existing global clustering and reconstruction stages.

`src/pipeline/tests.rs` adds window and full-pipeline source-versus-slice parity coverage. The changes bound memory use; they do not intentionally change speaker diarization behavior, model weights, clustering policy, or reconstruction algorithms. CPU whole-pipeline parity is an ignored explicit test; CoreML and accelerator qualification are backend-specific and require their fixtures and matching hardware. The application adapter remains in RecapIt's `frontend/src-tauri/src/audio/file_transcriber.rs`.

## Numerical fixtures

The fork retains the 23 upstream numerical fixtures and adds offline fixture provenance documentation and verification scripts under `fixtures/` and `scripts/`. See `fixtures/README.md` and `fixtures/UPSTREAM_PROVENANCE.json`. Model-dependent tests still require their documented model and audio assets.
