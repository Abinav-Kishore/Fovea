"""Tests for the multi-dataset abstraction (src/datasets/).

Runs against REAL data on disk for both datasets -- no mocking, matching the
rest of this repo's "nothing fabricated" convention. RELLIS-3D tests are
skipped (not failed) if the RELLIS directories aren't present on this
machine, since RELLIS is a large (~9GB) optional download.

Run with: python -m unittest discover tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.datasets import available_datasets, get_adapter
from src.datasets.rellis_labels import LABEL_DICT as RELLIS_LABEL_DICT
from src.semantic_labels import LABEL_DICT as SEMANTICPOSS_LABEL_DICT

CFG = load_config("config.yaml")
_RELLIS_PRESENT = Path(CFG["datasets"]["RELLIS-3D"]["pointcloud_root"]).exists()


class TestDatasetRegistry(unittest.TestCase):
    def test_both_datasets_registered(self):
        self.assertEqual(set(available_datasets()), {"SemanticPOSS", "RELLIS-3D"})

    def test_default_dataset_is_semanticposs(self):
        self.assertEqual(CFG["datasets"]["default"], "SemanticPOSS")

    def test_unknown_dataset_raises(self):
        with self.assertRaises(ValueError):
            get_adapter("NotADataset", CFG)

    def test_ontologies_are_kept_separate(self):
        # Regression guard: RELLIS and SemanticPOSS must never share a LABEL_DICT.
        self.assertIsNot(RELLIS_LABEL_DICT, SEMANTICPOSS_LABEL_DICT)
        # SemanticPOSS's sparse id set and RELLIS's are different ontologies --
        # id 1 ("dirt") exists in RELLIS but not SemanticPOSS; id 7 ("car") exists
        # in SemanticPOSS but means "sky" in RELLIS. Cross-checking this pins down
        # that the two tables were not accidentally merged or aliased.
        self.assertEqual(RELLIS_LABEL_DICT[1], "dirt")
        self.assertNotIn(1, SEMANTICPOSS_LABEL_DICT)
        self.assertEqual(SEMANTICPOSS_LABEL_DICT[7], "car")
        self.assertEqual(RELLIS_LABEL_DICT[7], "sky")


class TestSemanticPOSSAdapterUnchanged(unittest.TestCase):
    """The adapter must reproduce the exact pre-existing SemanticPOSS
    behavior -- these numbers are cross-checked against README.md."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)

    def test_sequences(self):
        self.assertIn("00", self.adapter.list_sequences())

    def test_frame_count_matches_known_value(self):
        self.assertEqual(len(self.adapter.list_frames("00")), 488)

    def test_known_frame_point_count(self):
        frame = self.adapter.load_frame("00", "000000")
        self.assertEqual(frame.points.shape[0], 66658)
        self.assertTrue(frame.has_labels)

    def test_label_dict_is_semanticposs(self):
        self.assertEqual(self.adapter.label_dict(), SEMANTICPOSS_LABEL_DICT)

    def test_semantic_priority_table_matches_config(self):
        self.assertEqual(self.adapter.semantic_priority_table(), CFG["fovea"]["semantic_priority"])
        self.assertEqual(self.adapter.semantic_priority_table()["car"], 0.9)


@unittest.skipUnless(_RELLIS_PRESENT, "RELLIS-3D data not present on this machine")
class TestRellisDatasetDiscovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("RELLIS-3D", CFG)

    def test_all_five_sequences_present(self):
        self.assertEqual(self.adapter.list_sequences(), ["00000", "00001", "00002", "00003", "00004"])

    def test_frames_discovered_for_each_sequence(self):
        for seq in self.adapter.list_sequences():
            frames = self.adapter.list_frames(seq)
            self.assertGreater(len(frames), 0, f"sequence {seq} should have usable frames")

    def test_only_frames_with_both_bin_and_label_are_listed(self):
        import numpy as np

        rcfg = CFG["datasets"]["RELLIS-3D"]
        bin_dir = Path(rcfg["pointcloud_root"]) / "00000" / rcfg["pointcloud_subdir"]
        label_dir = Path(rcfg["label_root"]) / "00000" / rcfg["label_subdir"]
        bin_stems = {p.stem for p in bin_dir.glob("*.bin")}
        label_stems = {p.stem for p in label_dir.glob("*.label")}
        expected = sorted(bin_stems & label_stems)
        self.assertEqual(self.adapter.list_frames("00000"), expected)
        # sanity: this repo's real download has matching counts, but the
        # adapter's intersection logic is what's actually under test above.
        del np


class TestRellisBinAndLabelParsing(unittest.TestCase):
    """Verifies actual byte-level parsing against real files, per sequence."""

    @classmethod
    def setUpClass(cls):
        if not _RELLIS_PRESENT:
            raise unittest.SkipTest("RELLIS-3D data not present on this machine")
        cls.adapter = get_adapter("RELLIS-3D", CFG)

    def test_bin_is_float32_xyzi_across_all_sequences(self):
        for seq in self.adapter.list_sequences():
            frames = self.adapter.list_frames(seq)
            frame = self.adapter.load_frame(seq, frames[0])
            self.assertEqual(frame.points.dtype, __import__("numpy").float32)
            self.assertEqual(frame.points.shape[1], 4)  # x, y, z, intensity
            self.assertGreater(frame.points.shape[0], 0)

    def test_point_count_equals_label_count_across_all_sequences(self):
        for seq in self.adapter.list_sequences():
            frames = self.adapter.list_frames(seq)
            frame = self.adapter.load_frame(seq, frames[0])
            self.assertTrue(frame.has_labels)
            self.assertEqual(frame.points.shape[0], frame.semantic_labels.shape[0])
            self.assertEqual(frame.points.shape[0], frame.instance_labels.shape[0])

    def test_semantic_ids_are_within_official_ontology(self):
        """Every semantic id actually observed in real label files must be a
        key in the official RELLIS LABEL_DICT (not "unknown") -- this is the
        cross-check that the fetched ontology matches the real data, not a
        coincidence."""
        import numpy as np

        for seq in self.adapter.list_sequences():
            frames = self.adapter.list_frames(seq)
            frame = self.adapter.load_frame(seq, frames[len(frames) // 2])  # a mid-sequence frame
            observed_ids = set(np.unique(frame.semantic_labels).tolist())
            unknown = observed_ids - set(RELLIS_LABEL_DICT.keys())
            self.assertEqual(
                unknown, set(),
                f"seq {seq}: observed semantic id(s) {unknown} not in the official RELLIS LABEL_DICT",
            )

    def test_mismatched_point_and_label_count_raises(self):
        """load_frame must refuse misaligned data, not truncate/pad silently
        -- mirrors data_loader.load_frame's contract for SemanticPOSS."""
        import numpy as np
        import tempfile

        rcfg = dict(CFG["datasets"]["RELLIS-3D"])
        with tempfile.TemporaryDirectory() as tmp:
            seq = "99999"
            bin_dir = Path(tmp) / "pc" / seq / rcfg["pointcloud_subdir"]
            label_dir = Path(tmp) / "lbl" / seq / rcfg["label_subdir"]
            bin_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            np.zeros(40, dtype=np.float32).tofile(bin_dir / "000000.bin")  # 10 points
            np.zeros(5, dtype=np.uint32).tofile(label_dir / "000000.label")  # only 5 labels

            bad_rcfg = dict(rcfg, pointcloud_root=str(Path(tmp) / "pc"), label_root=str(Path(tmp) / "lbl"))
            from src.datasets import rellis

            with self.assertRaises(ValueError):
                rellis.load_frame(bad_rcfg, seq, "000000")


@unittest.skipUnless(_RELLIS_PRESENT, "RELLIS-3D data not present on this machine")
class TestRellisEngineIntegration(unittest.TestCase):
    """End-to-end: real RELLIS frame -> unmodified FOVEA engine."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("RELLIS-3D", CFG)

    def _effective_cfg(self):
        import copy

        cfg = copy.deepcopy(CFG)
        cfg["fovea"]["semantic_priority"] = self.adapter.semantic_priority_table()
        cfg["fovea"]["dynamic_classes"] = list(self.adapter.dynamic_classes())
        return cfg

    def test_fovea_map_builds_on_real_rellis_frame(self):
        from src.fovea_allocator import build_fovea_map

        frame = self.adapter.load_frame("00000", "000100")
        cfg = self._effective_cfg()
        cells, accounting = build_fovea_map(
            frame.points, frame.semantic_labels, cfg, label_dict=self.adapter.label_dict()
        )
        self.assertGreater(len(cells), 0)
        self.assertEqual(accounting["method"], "fovea")
        names = {c.majority_class_name for c in cells}
        self.assertTrue(names.issubset(set(RELLIS_LABEL_DICT.values()) | {"unknown"}))

    def test_person_class_gets_high_information_score(self):
        """Cross-check the dataset-specific priority table actually takes
        effect: RELLIS's 'person' priority (1.0, config.yaml) should surface
        as a high-information cell, same behavior class as SemanticPOSS's
        '1 person' cells."""
        from src.fovea_allocator import build_fovea_map

        frame = self.adapter.load_frame("00000", "000100")
        cfg = self._effective_cfg()
        cells, _ = build_fovea_map(frame.points, frame.semantic_labels, cfg, label_dict=self.adapter.label_dict())
        person_cells = [c for c in cells if c.majority_class_name == "person"]
        self.assertGreater(len(person_cells), 0, "expected frame to contain a 'person' cell")
        self.assertTrue(any(c.information_score > 0.5 for c in person_cells))


if __name__ == "__main__":
    unittest.main()
