# speakrs v0.5.0 numerical test fixtures

These 23 NPY files are byte-for-byte upstream fixtures from `avencera/speakrs` tag `v0.5.0`, commit `96f3cec0756bc9df7f830a9ed31e782ddd445edd`. Their SHA-256 digests and Git blob IDs are recorded in `UPSTREAM_PROVENANCE.json`; `scripts/verify_fixture_provenance.py` checks both for every file. The package retains its existing Apache-2.0 `LICENSE`.

Twenty-two files are referenced by the current Rust tests, including the `powerset_mapping_{nc}_{ms}.npy` family. `pipeline_speaker_counting_data.npy` is preserved from upstream but has no current source reference. No audio, model weights, tokens, or generated substitutes are included. The PLDA tests that read `fixtures/models` and model-backed/audio tests still need separate inputs and are not part of the fixture-only test result below.

## Reproduce the isolated package checks

The RecapIt workspace lists `recapit`, `silero`, and `llama-helper` as members; patched `speakrs` is a dependency, not a member. Consequently, `-p speakrs` does not select a workspace member, and the workspace also rejects `--features default-linalg` when targeting this nonmember. Test the published package in an isolated temporary directory instead:

1. Copy this crate source and the RecapIt workspace `Cargo.lock` into a temporary directory outside the workspace.
2. In that isolated source copy, derive a standalone lockfile with `cargo +1.89.0 update --offline --workspace`. This resolves the crate's graph against the local Cargo cache; it does not edit the repository lockfile.
3. From the isolated source copy, create the package with `cargo +1.89.0 package --locked --offline --no-verify --no-default-features --features default-linalg`, then extract the `.crate` archive into a fresh directory. Make the derived standalone lockfile available as `Cargo.lock` in the extracted crate.
4. From the extracted crate root, run the four exact library tests below with the derived lockfile. `default-linalg` selects the BLAS backend required by `src/linalg.rs`; `--no-default-features` keeps the online model-download feature disabled.

```sh
for test in \
  powerset::tests::mapping_matrices_match_fixtures \
  powerset::tests::hard_decode_matches_fixture \
  clustering::ahc::tests::cluster_matches_python_fixture \
  clustering::vbx::tests::cluster_vbx_matches_python_fixture; do
  cargo +1.89.0 test --locked --offline --no-default-features --features default-linalg --lib "$test" -- --exact
done
```

`cargo --offline` prevents Cargo from fetching registry dependencies, but build scripts may still attempt external downloads. The recorded run used a network-denied execution profile and locally prepared native build dependencies; an offline Cargo flag alone is not a network sandbox. The package contained 97 members, including all 23 fixtures (269,131 bytes), and no models or audio. All four listed data-only tests passed once each against that packaged crate. This result does not cover PLDA/model tests, audio, inference, or native acceptance. The independent receipt pins and exact package/lockfile hashes are in the external review packet; no machine-specific cache paths are part of this guide.

Run `python3 scripts/verify_fixture_provenance.py` from the crate root to verify fixture bytes without compiling or loading models.
