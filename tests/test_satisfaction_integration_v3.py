import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.strategy.experiments import export_experiment_metadata
from src.v3.execution import has_executable_v3_changes


class SatisfactionIntegrationV3Tests(unittest.TestCase):
    def test_satisfaction_ranking_change_forces_downstream_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            highlights = Path(tmp)
            (highlights / "video.json").write_text(
                json.dumps(
                    [
                        {
                            "v3": {
                                "hook_mode": "NATIVE",
                                "no_transformation_needed": True,
                                "satisfaction_ranking_changed": True,
                            },
                            "satisfaction": {
                                "status": "available",
                                "end_adjustment_applied": False,
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with patch("src.v3.execution.HIGHLIGHTS_DIR", highlights):
                self.assertTrue(has_executable_v3_changes("video"))

    def test_satisfaction_end_change_forces_downstream_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            highlights = Path(tmp)
            (highlights / "video.json").write_text(
                json.dumps(
                    [
                        {
                            "v3": {
                                "hook_mode": "NATIVE",
                                "no_transformation_needed": True,
                                "satisfaction_ranking_changed": False,
                            },
                            "satisfaction": {
                                "status": "available",
                                "end_adjustment_applied": True,
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with patch("src.v3.execution.HIGHLIGHTS_DIR", highlights):
                self.assertTrue(has_executable_v3_changes("video"))

    def test_satisfaction_metadata_is_exported_for_learning_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights = root / "highlights"
            output = root / "output"
            highlights.mkdir()
            (highlights / "video.json").write_text(
                json.dumps(
                    [
                        {
                            "start": 10.0,
                            "end": 28.0,
                            "duration": 18.0,
                            "format_id": "entertainment_unexpected_reaction",
                            "format_version": 1,
                            "format_fit_score": 86,
                            "content_opportunity_score": 82,
                            "strategic_gate": "PASS",
                            "v3": {
                                "hook_mode": "RECONSTRUCTED",
                                "hook_score_v3": 92,
                                "originality_score": 81,
                            },
                            "satisfaction": {
                                "status": "available",
                                "viewer_satisfaction_score": 90,
                                "payoff_score": 94,
                                "expectation_match_score": 91,
                                "context_independence_score": 88,
                                "clarity_score": 92,
                                "emotional_completeness_score": 89,
                                "value_density_score": 87,
                                "ending_quality_score": 95,
                            },
                            "quality_gate": {"status": "PASS", "passed": True},
                            "final_score": {"score": 89.4},
                            "satisfaction_rank": 1,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with (
                patch("src.strategy.experiments.HIGHLIGHTS_DIR", highlights),
                patch("src.strategy.experiments.OUTPUT_DIR", output),
            ):
                path = export_experiment_metadata("video")

            payload = json.loads(path.read_text(encoding="utf-8"))
            item = payload["shorts"][0]
            self.assertEqual(item["viewer_satisfaction_score"], 90)
            self.assertEqual(item["payoff_score"], 94)
            self.assertEqual(item["context_independence_score"], 88)
            self.assertEqual(item["final_score"], 89.4)
            self.assertEqual(item["quality_gate"], "PASS")


if __name__ == "__main__":
    unittest.main()
