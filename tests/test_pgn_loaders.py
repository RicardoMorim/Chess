import tempfile
import textwrap
import unittest
from pathlib import Path

from train.core.data import (
    discover_pgn_files,
    load_pgn_games_from_directory,
    load_puzzle_examples_from_directory,
    load_training_examples_from_chess_pgns,
)


GAME_PGN = textwrap.dedent(
    """\
    [Event "Sample Game"]
    [Site "Local"]
    [Date "2026.05.29"]
    [Round "1"]
    [White "Alpha"]
    [Black "Beta"]
    [Result "1-0"]

    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
    """
)

PUZZLE_PGN = textwrap.dedent(
    """\
    [Event "Sample Puzzle"]
    [Site "Local"]
    [Date "2026.05.29"]
    [Round "1"]
    [White "Mate in one"]
    [Black "Defender"]
    [Result "1-0"]

    1. e4 e5 2. Qh5 Nc6 3. Qxe5+ Nxe5 1-0
    """
)


class PGNLoaderTests(unittest.TestCase):
    def _write(self, root: Path, relative: str, content: str) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_discover_and_load_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "pros/sample.pgn", GAME_PGN)
            self._write(root, "high_elo/second.pgn", GAME_PGN)

            files = discover_pgn_files(root_dir=root, subdirs=("pros", "high_elo"))
            self.assertEqual(len(files), 2)

            games = load_pgn_games_from_directory(root_dir=root, subdirs=("pros", "high_elo"))
            self.assertEqual(len(games), 2)
            self.assertEqual(games[0].headers["Result"], "1-0")

    def test_load_puzzle_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "puzzles/sample.pgn", PUZZLE_PGN)

            puzzles = load_puzzle_examples_from_directory(root_dir=root, subdirs=("puzzles",))
            self.assertEqual(len(puzzles), 1)
            fen, move, value, category = puzzles[0]
            self.assertTrue(fen)
            self.assertTrue(move)
            self.assertAlmostEqual(value, 1.0)
            self.assertEqual(category, "mate_in_one")

    def test_combined_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "pros/sample.pgn", GAME_PGN)
            self._write(root, "puzzles/sample.pgn", PUZZLE_PGN)

            bundle = load_training_examples_from_chess_pgns(root_dir=root)
            self.assertEqual(len(bundle["games"]), 1)
            self.assertEqual(len(bundle["puzzles"]), 1)


if __name__ == "__main__":
    unittest.main()
