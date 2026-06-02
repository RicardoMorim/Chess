import json
import tempfile
import unittest
from pathlib import Path

from train.league.evolution_logger import EvolutionLogger


class EvolutionLoggerTests(unittest.TestCase):
    def test_append_round_writes_markdown_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = EvolutionLogger(tmp)
            summary = {
                "elapsed_seconds": 12.3,
                "variants": {
                    "baseline": {"fair_score": 0.61, "fair_elo": 120.0, "gpu_avg_batch": 16.4, "gpu_flush_wait": 8},
                    "attack": {"fair_score": 0.50, "fair_elo": 0.0, "gpu_avg_batch": 12.2, "gpu_flush_wait": 10},
                    "est": {"fair_score": 0.44, "fair_elo": -80.0, "gpu_avg_batch": 14.8, "gpu_flush_wait": 9},
                },
            }

            logger.append_round(3, summary, round_time_seconds=9.5, note="test note")
            logger.append_note("extra note")

            md_path = Path(tmp) / "league_evolution.md"
            jsonl_path = Path(tmp) / "league_evolution.jsonl"

            self.assertTrue(md_path.exists())
            self.assertTrue(jsonl_path.exists())

            markdown = md_path.read_text(encoding="utf-8")
            # Table row: 8 core columns only (GPU stats are in dashboard summary below)
            self.assertIn(
                "| 3 | 9.5 | 0.610 | 0.500 | 0.440 | 120.000 | 0.000 | -80.000 |",
                markdown,
            )
            # Dashboard summary line with GPU batch stats
            self.assertIn(
                "avg_batch: b=16.40 a=12.20 e=14.80",
                markdown,
            )
            self.assertIn("- extra note", markdown)

            records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["round"], 3)
            self.assertEqual(records[0]["note"], "test note")


if __name__ == "__main__":
    unittest.main()
