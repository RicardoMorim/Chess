"""
Puzzle sidecar builder (Fase 4b).

The cached puzzle tensor files (``train/cache/puzzle_tensors/*.pkl``) only
store ``(tensor, policy, value)`` triples — they do **not** preserve the
original FEN, the solution line, the rating, or the themes. That makes
"puzzle drills" in spectate mode impossible to run from cache alone.

This module writes a lightweight sidecar mapping
``puzzle_id -> {fen, solution_moves, rating, themes, opening_tags}`` to
``train/cache/puzzles_meta.pkl`` (or whatever ``--output`` points at).

Build it once after downloading a new puzzle CSV; the spectate worker
will load it lazily on the first drill request and reuse it forever.

CLI:
    python -m league.puzzle_sidecar \\
        --csv train/chess_pgns/puzzles/lichess_db_puzzle.csv \\
        --output train/cache/puzzles_meta.pkl \\
        [--max-rows 50000]   # for tests / quick builds
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SIDECAR_SCHEMA_VERSION = 1


def _cache_key(csv_path: str) -> str:
    """Stable key derived from absolute path + size + mtime."""
    p = Path(csv_path).resolve()
    st = p.stat()
    raw = f"{p}|{st.st_size}|{int(st.st_mtime)}".encode()
    return hashlib.md5(raw).hexdigest()[:12]


def build_puzzle_sidecar(
    csv_path: str,
    output_path: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, dict]:
    """Parse the Lichess puzzle CSV and return a ``{puzzle_id: meta}`` dict.

    Args:
        csv_path: Path to ``lichess_db_puzzle.csv``.
        output_path: If set, write a pickle to this path. Defaults to
            ``<repo>/train/cache/puzzles_meta.pkl``.
        max_rows: Optional cap (for tests).

    Returns:
        Mapping ``puzzle_id -> {fen, solution_moves, rating, themes, opening_tags}``.
    """
    csv_path = str(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Puzzle CSV not found: {csv_path}")

    if output_path is None:
        here = Path(__file__).resolve().parent.parent  # train/
        cache_dir = here / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(cache_dir / "puzzles_meta.pkl")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building puzzle sidecar from {csv_path} -> {output_path}")
    t0 = time.time()
    puzzles: Dict[str, dict] = {}
    skipped = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            try:
                pid = row.get("PuzzleId") or row.get("puzzle_id")
                fen = row.get("FEN") or row.get("fen")
                moves_raw = row.get("Moves") or row.get("moves") or ""
                rating_raw = row.get("Rating") or row.get("rating") or "0"
                themes_raw = row.get("Themes") or row.get("themes") or ""
                opening_raw = row.get("OpeningTags") or row.get("opening_tags") or ""
                if not pid or not fen:
                    skipped += 1
                    continue
                moves = moves_raw.split()
                if not moves:
                    skipped += 1
                    continue
                try:
                    rating = int(float(rating_raw))
                except (ValueError, TypeError):
                    rating = 0
                themes = [t for t in themes_raw.split() if t]
                opening_tags = [t for t in opening_raw.split() if t]
                puzzles[pid] = {
                    "fen": fen,
                    "solution_moves": moves,
                    "rating": rating,
                    "themes": themes,
                    "opening_tags": opening_tags,
                }
            except Exception as e:
                logger.debug(f"Skipping puzzle {i}: {e}")
                skipped += 1

    payload = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "source_csv": str(Path(csv_path).resolve()),
        "source_key": _cache_key(csv_path),
        "built_at": time.time(),
        "count": len(puzzles),
        "puzzles": puzzles,
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        f"Puzzle sidecar built: {len(puzzles)} puzzles, "
        f"{skipped} skipped, {time.time() - t0:.1f}s, "
        f"size={os.path.getsize(output_path) / 1e6:.1f}MB"
    )
    return puzzles


def load_puzzle_sidecar(
    path: Optional[str] = None,
    csv_path: Optional[str] = None,
    auto_build: bool = True,
    max_rows: Optional[int] = None,
) -> Optional[Dict[str, dict]]:
    """Load the sidecar from disk. Returns the inner ``puzzles`` dict, or ``None``.

    If ``path`` is missing and ``auto_build`` is True, attempts to build
    it from ``csv_path`` (default: the first ``lichess_db_puzzle.csv``
    under ``train/chess_pgns/puzzles/``). Returns ``None`` if everything
    fails — callers should fall back to a "no puzzles" error event.
    """
    if path is None:
        here = Path(__file__).resolve().parent.parent
        path = str(here / "cache" / "puzzles_meta.pkl")

    if not os.path.exists(path):
        if not auto_build:
            return None
        if csv_path is None:
            csv_path = _find_default_csv()
            if csv_path is None:
                logger.warning("No puzzle CSV found; cannot build sidecar")
                return None
        try:
            build_puzzle_sidecar(csv_path, output_path=path, max_rows=max_rows)
        except Exception as e:
            logger.error(f"Failed to build puzzle sidecar: {e}")
            return None

    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load puzzle sidecar {path}: {e}")
        return None
    if not isinstance(payload, dict) or "puzzles" not in payload:
        logger.error(f"Sidecar {path} has unexpected schema")
        return None
    return payload["puzzles"]


def _find_default_csv() -> Optional[str]:
    """Look for a Lichess puzzle CSV in the standard location."""
    here = Path(__file__).resolve().parent.parent
    candidates = [
        here / "chess_pgns" / "puzzles" / "lichess_db_puzzle.csv",
        here / "chess_pgns" / "puzzles" / "lichess_db_puzzle.csv.zst",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    puz_dir = here / "chess_pgns" / "puzzles"
    if puz_dir.exists():
        for p in puz_dir.glob("lichess_db_puzzle*.csv*"):
            return str(p)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the puzzle metadata sidecar (puzzle_id -> {fen, solution, ...})"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to lichess_db_puzzle.csv (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output pickle path (default: train/cache/puzzles_meta.pkl)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap on number of rows to parse (for tests / quick smoke)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    csv_path = args.csv or _find_default_csv()
    if not csv_path:
        print("ERROR: no puzzle CSV found. Pass --csv path/to/lichess_db_puzzle.csv",
              file=sys.stderr)
        return 2
    puzzles = build_puzzle_sidecar(
        csv_path=csv_path, output_path=args.output, max_rows=args.max_rows
    )
    print(f"OK: wrote {len(puzzles)} puzzles to {args.output or 'train/cache/puzzles_meta.pkl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
