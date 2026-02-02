#!/usr/bin/env python3
"""
League Training System - Setup Validation

Checks if all components are correctly installed and configured.
Run this before starting league/main.py to catch issues early.

Usage:
    python validate_setup.py
"""

import sys
from pathlib import Path

def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description}")
        return False

def check_module_imports() -> bool:
    """Check if key modules can be imported."""
    print("\n" + "="*60)
    print("CHECKING PYTHON IMPORTS")
    print("="*60)
    
    all_ok = True
    
    # Core dependencies
    modules = [
        ("torch", "PyTorch"),
        ("chess", "python-chess"),
        ("numpy", "NumPy"),
    ]
    
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} - Install with: pip install {module_name}")
            all_ok = False
    
    # Project modules
    print("\nProject modules:")
    try:
        from train.models import create_model
        print(f"✓ train.models.create_model")
    except ImportError as e:
        print(f"✗ train.models - {e}")
        all_ok = False
    
    try:
        from train.mcts import MCTS
        print(f"✓ train.mcts.MCTS")
    except ImportError as e:
        print(f"✗ train.mcts - {e}")
        all_ok = False
    
    try:
        from train.data import board_to_tensor
        print(f"✓ train.data.board_to_tensor")
    except ImportError as e:
        print(f"✗ train.data - {e}")
        all_ok = False
    
    # League modules
    print("\nLeague system modules:")
    try:
        from league.replay_buffer import ReplayBuffer
        print(f"✓ league.replay_buffer.ReplayBuffer")
    except ImportError as e:
        print(f"✗ league.replay_buffer - {e}")
        all_ok = False
    
    try:
        from league.league_trainer import LeagueTrainer
        print(f"✓ league.league_trainer.LeagueTrainer")
    except ImportError as e:
        print(f"✗ league.league_trainer - {e}")
        all_ok = False
    
    try:
        from league.monitoring import MetricsCollector
        print(f"✓ league.monitoring.MetricsCollector")
    except ImportError as e:
        print(f"✗ league.monitoring - {e}")
        all_ok = False
    
    return all_ok

def check_directory_structure() -> bool:
    """Check if required directories exist."""
    print("\n" + "="*60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*60)
    
    all_ok = True
    
    dirs = [
        ("league", "League training system"),
        ("bootstrap", "Bootstrap training module"),
        ("train", "Existing training code"),
        ("docs", "Documentation"),
    ]
    
    for dir_name, description in dirs:
        path = Path(dir_name)
        if path.exists() and path.is_dir():
            print(f"✓ {description} ({dir_name}/)")
        else:
            print(f"✗ {description} ({dir_name}/) - Directory not found")
            all_ok = False
    
    return all_ok

def check_critical_files() -> bool:
    """Check if critical files exist."""
    print("\n" + "="*60)
    print("CHECKING CRITICAL FILES")
    print("="*60)
    
    all_ok = True
    
    files = [
        ("league/main.py", "League training entry point"),
        ("league/__init__.py", "League package initialization"),
        ("league/league_trainer.py", "Main orchestrator"),
        ("league/replay_buffer.py", "Replay buffer"),
        ("league/self_play_worker.py", "Self-play worker"),
        ("league/evaluator.py", "Evaluator"),
        ("league/monitoring.py", "Monitoring system"),
        ("docs/README.md", "Documentation index"),
        ("docs/QUICKSTART.md", "Quick start guide"),
        ("docs/ARCHITECTURE.md", "Architecture documentation"),
        ("train/models.py", "Model definitions"),
        ("train/mcts.py", "MCTS implementation"),
    ]
    
    for file_path, description in files:
        all_ok = check_file_exists(file_path, description) and all_ok
    
    return all_ok

def check_pytorch_cuda() -> bool:
    """Check PyTorch and CUDA setup."""
    print("\n" + "="*60)
    print("CHECKING PYTORCH & CUDA")
    print("="*60)
    
    try:
        import torch
        
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✓ CUDA available ({device_count} device{'s' if device_count > 1 else ''})")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1e9
                print(f"  Device {i}: {props.name} ({vram_gb:.1f} GB VRAM)")
            
            return True
        else:
            print(f"⚠ CUDA not available - will use CPU (training will be slow)")
            return True  # Not a hard failure
    
    except ImportError:
        print(f"✗ PyTorch not installed")
        return False

def test_mcts_integration() -> bool:
    """Test that MCTS works with model."""
    print("\n" + "="*60)
    print("TESTING MCTS INTEGRATION")
    print("="*60)
    
    try:
        import torch
        import chess
        from train.models import create_model
        from train.mcts import MCTS
        
        print("Creating test model...")
        model = create_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"✓ Model created and moved to {device}")
        
        print("Creating MCTS searcher...")
        mcts = MCTS(model, device=device, num_visits=10)  # Only 10 visits for speed
        print(f"✓ MCTS searcher created")
        
        print("Running single MCTS search...")
        board = chess.Board()
        policy, move = mcts.search(board)
        print(f"✓ MCTS search successful")
        print(f"  Policy shape: {policy.shape}")
        print(f"  Move: {move}")
        
        return True
    
    except Exception as e:
        print(f"✗ MCTS integration test failed: {e}")
        return False

def main():
    """Run all validation checks."""
    
    print("\n" + "="*60)
    print("LEAGUE TRAINING SYSTEM - SETUP VALIDATION")
    print("="*60 + "\n")
    
    results = {
        "Directory structure": check_directory_structure(),
        "Critical files": check_critical_files(),
        "Python imports": check_module_imports(),
        "PyTorch & CUDA": check_pytorch_cuda(),
        "MCTS integration": test_mcts_integration(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_ok = all(results.values())
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    print("="*60)
    
    if all_ok:
        print("\n✓ ALL CHECKS PASSED - System is ready!")
        print("\nYou can now run:")
        print("    python league/main.py")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED - See above for details")
        print("\nFix the issues listed above before running league/main.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
