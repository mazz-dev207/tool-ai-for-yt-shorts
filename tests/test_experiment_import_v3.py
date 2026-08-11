import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.strategy.experiments import import_performance_metadata


class ExperimentImportV3Tests(unittest.TestCase):
    def test_import_associates_metrics_by_short_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            aggregate = output / "debug" / "video" / "experiment_metadata.json"
            aggregate.parent.mkdir(parents=True)
            aggregate.write_text(
                json.dumps(
                    {
                        "video": "video",
                        "shorts": [
                            {
                                "short_id": "video_clip_1",
                                "format_id": "gaming_clutch_comeback",
                                "performance": {
                                    "views": None,
                                    "viewed_vs_swiped": None,
                                    "average_view_duration": None,
                                    "average_percentage_viewed": None,
                                    "likes": None,
                                    "comments": None,
                                    "shares": None,
                                    "subscribers_gained": None,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            imported = root / "performance.json"
            imported.write_text(
                json.dumps(
                    {
                        "shorts": [
                            {
                                "short_id": "video_clip_1",
                                "views": 4200,
                                "viewed_vs_swiped": 72.5,
                                "average_view_duration": 14.2,
                                "average_percentage_viewed": 88.1,
                                "likes": 310,
                                "comments": 22,
                                "shares": 17,
                                "subscribers_gained": 9,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch("src.strategy.experiments.OUTPUT_DIR", output):
                result = import_performance_metadata("video", imported)

            saved = json.loads(result.read_text(encoding="utf-8"))
            metrics = saved["shorts"][0]["performance"]
            self.assertEqual(metrics["views"], 4200)
            self.assertEqual(metrics["viewed_vs_swiped"], 72.5)
            self.assertEqual(metrics["average_percentage_viewed"], 88.1)
            self.assertEqual(metrics["subscribers_gained"], 9)
            self.assertEqual(saved["performance_import"]["matched_shorts"], 1)
            self.assertTrue(
                (output / "debug" / "video" / "clip_1" / "experiment_metadata.json").exists()
            )

    def test_negative_or_invalid_metrics_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            aggregate = output / "debug" / "video" / "experiment_metadata.json"
            aggregate.parent.mkdir(parents=True)
            aggregate.write_text(
                json.dumps(
                    {
                        "video": "video",
                        "shorts": [
                            {
                                "short_id": "video_clip_1",
                                "performance": {"views": 100, "likes": 5},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            imported = root / "performance.json"
            imported.write_text(
                json.dumps(
                    [
                        {
                            "short_id": "video_clip_1",
                            "views": -10,
                            "likes": "not-a-number",
                            "comments": 3,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with patch("src.strategy.experiments.OUTPUT_DIR", output):
                result = import_performance_metadata("video", imported)

            saved = json.loads(result.read_text(encoding="utf-8"))
            metrics = saved["shorts"][0]["performance"]
            self.assertEqual(metrics["views"], 100)
            self.assertEqual(metrics["likes"], 5)
            self.assertEqual(metrics["comments"], 3)


if __name__ == "__main__":
    unittest.main()