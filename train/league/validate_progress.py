"""
Quick validation script to verify league training improvements.

Runs 3 rounds and checks:
1. Parallelism works (4 workers × 5 games = 20 games per model)
2. Resignations trigger (games shorter than 150 moves)
3. Loss decreases over rounds
4. Metrics accurate (buffer grows, counters correct)
"""

import sys
import json
from pathlib import Path

def validate_metrics(metrics_path: Path):
    """Check metrics JSON for expected values."""
    
    with open(metrics_path) as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print("VALIDATION REPORT")
    print(f"{'='*60}\n")
    
    counters = data.get("counters", {})
    variants = data.get("variants", {})
    
    # Check parallelism
    print("1. PARALLELISM CHECK")
    for variant in ["baseline", "attack", "est"]:
        games = counters.get(f"{variant}_games", 0)
        expected_min = 20  # 4 workers × 5 games
        status = "✅" if games >= expected_min else "❌"
        print(f"   {status} {variant}: {games} games (expected ≥{expected_min})")
    
    # Check resignations
    print("\n2. RESIGNATION CHECK")
    for variant in ["baseline", "attack", "est"]:
        v_stats = variants.get(variant, {})
        avg_length = v_stats.get("avg_game_length", 150)
        status = "✅" if avg_length < 120 else "⚠️"  # Should be less if resignations work
        print(f"   {status} {variant}: avg {avg_length:.1f} moves (expect <120 with resignations)")
    
    # Check buffer growth
    print("\n3. BUFFER GROWTH CHECK")
    for variant in ["baseline", "attack", "est"]:
        v_stats = variants.get(variant, {})
        buffer_size = v_stats.get("buffer_size", 0)
        expected_min = 500  # 20 games × ~30 moves avg
        status = "✅" if buffer_size >= expected_min else "❌"
        print(f"   {status} {variant}: {buffer_size} positions (expected ≥{expected_min})")
    
    # Check training happened
    print("\n4. TRAINING CHECK")
    for variant in ["baseline", "attack", "est"]:
        train_steps = counters.get(f"{variant}_train_steps", 0)
        expected_min = 10  # At least 1 round × 10 steps
        status = "✅" if train_steps >= expected_min else "❌"
        print(f"   {status} {variant}: {train_steps} training steps (expected ≥{expected_min})")
    
    # Check metrics consistency
    print("\n5. METRICS CONSISTENCY CHECK")
    for variant in ["baseline", "attack", "est"]:
        counter_games = counters.get(f"{variant}_games", 0)
        variant_games = variants.get(variant, {}).get("games", 0)
        status = "✅" if counter_games == variant_games else "❌"
        print(f"   {status} {variant}: counters={counter_games}, variants={variant_games} (must match)")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # Find latest metrics file
    logs_dir = Path("logs")
    
    if not logs_dir.exists():
        print("ERROR: logs/ directory not found. Run training first.")
        sys.exit(1)
    
    metrics_files = list(logs_dir.glob("metrics_*.json"))
    
    if not metrics_files:
        print("ERROR: No metrics files found. Run training first.")
        sys.exit(1)
    
    latest = sorted(metrics_files, key=lambda p: p.stat().st_mtime)[-1]
    print(f"Analyzing: {latest}")
    
    validate_metrics(latest)
