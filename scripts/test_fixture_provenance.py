import json
import pathlib
import shutil
import tempfile
import unittest

from verify_fixture_provenance import FIXTURES, verify_fixture_tree

class FixtureProvenanceTests(unittest.TestCase):
    def test_pinned_fixture_tree_matches_upstream(self):
        result = verify_fixture_tree()
        self.assertEqual(result["count"], 23)
        self.assertEqual(result["bytes"], 269131)

    def test_mutated_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "fixtures"
            shutil.copytree(FIXTURES, target)
            victim = target / "powerset_mapping_2_1.npy"
            data = bytearray(victim.read_bytes())
            data[-1] ^= 1
            victim.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "fixture_sha256_mismatch"):
                verify_fixture_tree(target)

    def test_cargo_distribution_and_offline_command_bind_to_real_backend(self):
        crate = pathlib.Path(__file__).resolve().parents[1]
        cargo = (crate / "Cargo.toml").read_text()
        self.assertIn('"/fixtures/*.npy"', cargo)
        self.assertIn('"/fixtures/UPSTREAM_PROVENANCE.json"', cargo)
        self.assertIn('"/scripts/verify_fixture_provenance.py"', cargo)
        self.assertIn('"/scripts/test_fixture_provenance.py"', cargo)
        self.assertNotIn('"/fixtures/**"', cargo)
        self.assertIn('default-linalg = ["dep:ndarray-linalg-default"]', cargo)
        linalg = (crate / "src/linalg.rs").read_text()
        self.assertIn('speakrs requires a BLAS backend', linalg)
        self.assertIn('feature = "default-linalg"', linalg)
        guide = (crate / "fixtures/README.md").read_text()
        self.assertIn('--locked --offline --no-default-features --features default-linalg --lib', guide)
        self.assertNotIn('--features online', guide)
        self.assertIn('not a member', guide)
        self.assertIn('cargo +1.89.0 update --offline --workspace', guide)
        self.assertIn('cargo +1.89.0 package --locked --offline --no-verify --no-default-features --features default-linalg', guide)
        self.assertIn('build scripts may still attempt external downloads', guide)
        self.assertIn('four exact library tests', guide)
        for test_name in (
            'powerset::tests::mapping_matrices_match_fixtures',
            'powerset::tests::hard_decode_matches_fixture',
            'clustering::ahc::tests::cluster_matches_python_fixture',
            'clustering::vbx::tests::cluster_vbx_matches_python_fixture',
        ):
            self.assertIn(test_name, guide)

    def test_readme_links_to_the_pinned_upstream_guide_not_missing_local_file(self):
        readme = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text()
        self.assertIn("96f3cec0756bc9df7f830a9ed31e782ddd445edd/CONTRIBUTING.md", readme)
        self.assertNotIn("](CONTRIBUTING.md)", readme)

    def test_missing_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "fixtures"
            shutil.copytree(FIXTURES, target)
            (target / "pipeline_ahc_clusters.npy").unlink()
            with self.assertRaisesRegex(ValueError, "fixture_disk_file_set_invalid"):
                verify_fixture_tree(target)

if __name__ == "__main__":
    unittest.main()
