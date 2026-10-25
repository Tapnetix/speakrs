#!/usr/bin/env python3
"""Verify the vendored speakrs numerical fixture set against its pinned upstream manifest."""
import hashlib
import json
import pathlib
import stat
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
MANIFEST = FIXTURES / "UPSTREAM_PROVENANCE.json"
EXPECTED_COMMIT = "96f3cec0756bc9df7f830a9ed31e782ddd445edd"
EXPECTED_COUNT = 23
EXPECTED_BYTES = 269131
EXPECTED_ORPHAN = "pipeline_speaker_counting_data.npy"

def verify_fixture_tree(root=FIXTURES):
    root = pathlib.Path(root)
    manifest = json.loads((root / "UPSTREAM_PROVENANCE.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "speakrs-upstream-fixture-provenance-v1"
            or manifest.get("repository") != "https://github.com/avencera/speakrs"
            or manifest.get("tag") != "v0.5.0" or manifest.get("commit") != EXPECTED_COMMIT
            or manifest.get("fixtureCount") != EXPECTED_COUNT
            or manifest.get("totalBytes") != EXPECTED_BYTES
            or manifest.get("modelOrAudioAssetsIncluded") is not False):
        raise ValueError("fixture_manifest_identity_invalid")
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != EXPECTED_COUNT:
        raise ValueError("fixture_manifest_file_set_invalid")
    names = [row.get("path") for row in rows if isinstance(row, dict)]
    if len(names) != EXPECTED_COUNT or len(set(names)) != EXPECTED_COUNT:
        raise ValueError("fixture_manifest_file_set_invalid")
    if any(not isinstance(name, str) or pathlib.PurePosixPath(name).name != name or not name.endswith(".npy") for name in names):
        raise ValueError("fixture_manifest_path_invalid")
    disk_names = sorted(p.name for p in root.iterdir() if p.suffix == ".npy")
    if sorted(names) != disk_names:
        raise ValueError("fixture_disk_file_set_invalid")
    total = 0
    for row in rows:
        path = root / row["path"]
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size != row.get("bytes"):
            raise ValueError("fixture_file_identity_invalid")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row.get("sha256"):
            raise ValueError("fixture_sha256_mismatch")
        git_blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
        if git_blob != row.get("upstreamGitBlob"):
            raise ValueError("fixture_upstream_blob_mismatch")
        expected_referenced = row["path"] != EXPECTED_ORPHAN
        if row.get("referencedByCurrentRustTests") is not expected_referenced:
            raise ValueError("fixture_reference_classification_invalid")
        if row.get("upstreamUrl") != f"https://github.com/avencera/speakrs/blob/{EXPECTED_COMMIT}/fixtures/{row['path']}":
            raise ValueError("fixture_source_url_invalid")
        total += len(data)
    if total != EXPECTED_BYTES:
        raise ValueError("fixture_total_bytes_invalid")
    return {"count": EXPECTED_COUNT, "bytes": total, "commit": EXPECTED_COMMIT}

if __name__ == "__main__":
    try:
        print(json.dumps(verify_fixture_tree(), sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"fixture verification refused: {error}", file=sys.stderr)
        raise SystemExit(1)
