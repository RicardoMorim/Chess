"""
Test GPU batching activation in LeagueTrainer.
"""
import sys
from train.league.league_trainer import LeagueTrainer
from train.core.models import create_model

try:
    # Create trainer with GPU batching enabled
    trainer = LeagueTrainer(
        device="cpu",  # Test on CPU (GPU batching works with CPU too)
        use_gpu_batching=True
    )
    print("✓ LeagueTrainer initialized with use_gpu_batching=True")
    
    # Check that use_gpu_batching is actually set
    assert trainer.use_gpu_batching == True, "use_gpu_batching not set correctly"
    print("✓ GPU batching flag is set correctly")
    
    # Try to create a model
    model = create_model(variant="baseline", num_blocks=4, channels=64)
    print("✓ Model created successfully")
    
    print("\n✅ GPU batching activation test passed!")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
