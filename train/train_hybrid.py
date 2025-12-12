#!/usr/bin/env python3
"""
Hybrid Training Script - Alternates between Pro Games and Self-Play
====================================================================

This script implements a hybrid training approach:
- 70% time on professional games (learning patterns)
- 30% time on self-play (reinforcement learning)

Usage:
    python train_hybrid.py --model limited --cycles 10

Each cycle consists of:
    1. Train on pro games (batch of ~5000 games)
    2. Run self-play (generate 20 games, train 3 epochs)
    3. Test tactical accuracy
    4. Save checkpoint
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and stream output."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n⚠ Command failed with return code {process.returncode}")
            return False
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Stopping gracefully...")
        process.terminate()
        process.wait()
        return False
    except Exception as e:
        print(f"\n❌ Error running command: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Hybrid training: Pro games + Self-play")
    parser.add_argument("--model", default="limited", choices=["limited", "small", "medium", "big"],
                       help="Model size to train")
    parser.add_argument("--cycles", type=int, default=10,
                       help="Number of training cycles")
    parser.add_argument("--pro-iterations", type=int, default=3,
                       help="Pro game iterations per cycle")
    parser.add_argument("--selfplay-games", type=int, default=20,
                       help="Self-play games per cycle")
    parser.add_argument("--selfplay-iterations", type=int, default=3,
                       help="Self-play training iterations")
    parser.add_argument("--fast-mcts", action="store_true",
                       help="Use fast MCTS (less quality, faster)")
    parser.add_argument("--skip-selfplay", action="store_true",
                       help="Skip self-play (only pro games)")
    
    args = parser.parse_args()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║            HYBRID CHESS AI TRAINING                               ║
║  Combining Professional Games with Self-Play Reinforcement        ║
╚═══════════════════════════════════════════════════════════════════╝

Model: {args.model}
Total cycles: {args.cycles}

Per cycle:
  • Pro games: {args.pro_iterations} iterations (~{args.pro_iterations * 1500} games)
  • Self-play: {args.selfplay_games} games × {args.selfplay_iterations} training epochs
  • MCTS mode: {'Fast' if args.fast_mcts else 'Full quality'}

Starting in 3 seconds... (Ctrl+C to cancel)
""")
    
    time.sleep(3)
    
    for cycle in range(1, args.cycles + 1):
        print(f"\n\n")
        print(f"╔{'═'*68}╗")
        print(f"║  CYCLE {cycle}/{args.cycles}{' '*(58-len(str(cycle))-len(str(args.cycles)))}║")
        print(f"╚{'═'*68}╝")
        
        # Phase 1: Professional games training
        print(f"\n📚 Phase 1: Learning from professional games...")
        
        pro_cmd = f"python train.py pro --model {args.model}"
        
        if not run_command(pro_cmd, f"Training on professional games (Cycle {cycle})"):
            print("\n❌ Pro games training failed or was interrupted.")
            if input("Continue to self-play anyway? (y/n): ").lower() != 'y':
                print("Stopping training.")
                return 1
        
        # Phase 2: Self-play reinforcement (optional)
        if not args.skip_selfplay:
            print(f"\n🤖 Phase 2: Self-play reinforcement learning...")
            
            selfplay_cmd = (
                f"python train.py self-play {args.selfplay_games} {args.selfplay_iterations} "
                f"--model {args.model}"
            )
            
            if args.fast_mcts:
                selfplay_cmd += " --fast-mcts"
            
            if not run_command(selfplay_cmd, f"Self-play training (Cycle {cycle})"):
                print("\n❌ Self-play training failed or was interrupted.")
                if input("Continue to next cycle? (y/n): ").lower() != 'y':
                    print("Stopping training.")
                    return 1
        
        # Phase 3: Brief pause and status
        print(f"\n✅ Cycle {cycle}/{args.cycles} complete!")
        print(f"   Model checkpoint saved to: checkpoints_{args.model}/model_best.pt")
        
        if cycle < args.cycles:
            print(f"\n⏸  Pausing 5 seconds before next cycle...")
            time.sleep(5)
    
    print(f"""

╔═══════════════════════════════════════════════════════════════════╗
║                  🎉 TRAINING COMPLETE! 🎉                         ║
╚═══════════════════════════════════════════════════════════════════╝

All {args.cycles} training cycles completed successfully!

Final model saved to: train/checkpoints_{args.model}/model_best.pt

Next steps:
  1. Test the model: python ../Main.py
  2. Evaluate strength: Set up AI vs AI matches
  3. Continue training: Run this script again with more cycles

""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
