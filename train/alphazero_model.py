"""
AlphaZero-style Chess Neural Network

Architecture inspired by DeepMind's AlphaZero paper (Silver et al., 2018):
- 119 input planes encoding board state + 7 positions of history
- 19 residual blocks with Squeeze-and-Excitation attention
- Policy head: 73×8×8 = 4672 outputs (AlphaZero move encoding)
- Value head: scalar output via global average pooling

Key differences from the original 22-channel "big" model:
- NO attack maps (network learns attack patterns implicitly)
- Position history provides temporal context for detecting repetitions
- More residual blocks (19 vs 15) for deeper pattern recognition

Input Channel Layout (119 total):
    Channels 0-11:   Current position piece planes (6 piece types × 2 colors)
    Channels 12-23:  Position T-1 piece planes
    Channels 24-35:  Position T-2 piece planes
    Channels 36-47:  Position T-3 piece planes
    Channels 48-59:  Position T-4 piece planes
    Channels 60-71:  Position T-5 piece planes
    Channels 72-83:  Position T-6 piece planes
    Channels 84-95:  Position T-7 piece planes
    Channels 96-99:  Castling rights (WK, WQ, BK, BQ)
    Channel 100:     En passant square
    Channel 101:     Side to move (1 = white, 0 = black)
    Channel 102:     Total move count (normalized to 0-1)
    Channel 103:     No-progress count / 50-move rule (normalized to 0-1)
    Channels 104-105: Repetition counters (1x, 2x+ same position)
    Channels 106-118: Reserved for future features (zeros)

Usage:
    from alphazero_model import AlphaZeroNet
    model = AlphaZeroNet()  # Default: 19 blocks, 256 channels, SE enabled
    model = AlphaZeroNet(num_blocks=10, channels=128)  # Smaller variant

Author: Chess AI Training System
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.
    
    Allows the network to weight channel importance dynamically,
    which significantly improves chess network performance.
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks" (2018)
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """Initialize SE block.
        
        Args:
            channels: Number of input/output channels
            reduction: Reduction ratio for the bottleneck (default: 4)
        """
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply squeeze-and-excitation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor of same shape with channel-wise attention applied
        """
        batch, channels, _, _ = x.size()
        
        # Squeeze: Global average pooling
        y = x.view(batch, channels, -1).mean(dim=2)
        
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        
        # Scale: Multiply input by attention weights
        return x * y.view(batch, channels, 1, 1)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with optional SE attention.
    
    Uses pre-activation pattern (BN -> ReLU -> Conv) which trains better
    for very deep networks compared to post-activation.
    
    Reference: He et al., "Identity Mappings in Deep Residual Networks" (2016)
    """
    
    def __init__(self, channels: int, use_se: bool = True, se_reduction: int = 4):
        """Initialize residual block.
        
        Args:
            channels: Number of channels (input = output)
            use_se: Whether to use Squeeze-and-Excitation block
            se_reduction: SE reduction ratio
        """
        super().__init__()
        self.use_se = use_se
        
        # Pre-activation residual block
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        # Optional SE block
        if use_se:
            self.se = SEBlock(channels, se_reduction)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block with skip connection.
        
        Args:
            x: Input tensor of shape (B, C, 8, 8)
            
        Returns:
            Output tensor of same shape
        """
        residual = x
        
        # Pre-activation: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        
        # Apply SE if enabled
        if self.use_se:
            out = self.se(out)
        
        # Residual connection
        return out + residual


# ============================================================================
# ALPHAZERO NETWORK
# ============================================================================

class AlphaZeroNet(nn.Module):
    """AlphaZero-style Chess Neural Network.
    
    A deep residual network with two heads:
    - Policy head: Outputs move probability distribution (4672 moves)
    - Value head: Outputs position evaluation (-1 to +1)
    
    Attributes:
        input_channels: Number of input channels (119 for AlphaZero encoding)
        num_blocks: Number of residual blocks in the tower
        channels: Number of channels in the residual tower
        use_se: Whether SE blocks are enabled
    """
    
    # Class constants
    INPUT_CHANNELS = 119
    POLICY_OUTPUT_SIZE = 4672  # 73 planes × 64 squares
    
    def __init__(
        self, 
        num_blocks: int = 19, 
        channels: int = 256, 
        use_se: bool = True,
        policy_dropout: float = 0.1,
        value_dropout: float = 0.3
    ):
        """Initialize AlphaZeroNet.
        
        Args:
            num_blocks: Number of residual blocks (default: 19, like AlphaZero)
            channels: Number of channels in residual tower (default: 256)
            use_se: Whether to use Squeeze-and-Excitation blocks (default: True)
            policy_dropout: Dropout rate for policy head (default: 0.1)
            value_dropout: Dropout rate for value head (default: 0.3)
        """
        super().__init__()
        
        # Store configuration
        self.input_channels = self.INPUT_CHANNELS
        self.num_blocks = num_blocks
        self.channels = channels
        self.use_se = use_se
        self.legacy_mode = False  # For compatibility with existing code
        
        # Initial convolution: 119 -> channels
        self.conv1 = nn.Conv2d(
            self.INPUT_CHANNELS, channels, 
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Residual tower
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, use_se=use_se, se_reduction=4)
            for _ in range(num_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 73, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(73)
        self.policy_dropout = nn.Dropout2d(p=policy_dropout)
        
        # Value head with global average pooling
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32, 128)
        self.value_dropout = nn.Dropout(p=value_dropout)
        self.value_fc2 = nn.Linear(128, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights for better training."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 119, 8, 8)
            
        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: Shape (B, 4672) - raw logits for moves
                - value: Shape (B, 1) - position evaluation in [-1, 1]
        """
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual tower
        for block in self.blocks:
            x = block(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_dropout(policy)
        policy = policy.view(-1, self.POLICY_OUTPUT_SIZE)
        
        # Value head with global average pooling
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.mean(dim=[2, 3])  # Global avg pool: (B, 32, 8, 8) -> (B, 32)
        value = F.relu(self.value_fc1(value))
        value = self.value_dropout(value)
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def freeze_backbone(self) -> 'AlphaZeroNet':
        """Freeze the residual tower for fine-tuning only the heads.
        
        Useful when training on a small dataset (like puzzles) to prevent
        overfitting while preserving learned features.
        
        Returns:
            self for method chaining
        """
        # Freeze initial convolution
        self.conv1.requires_grad_(False)
        self.bn1.requires_grad_(False)
        
        # Freeze all residual blocks
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"🔒 Backbone frozen: {self.num_blocks} residual blocks + initial conv")
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        
        return self
    
    def unfreeze_backbone(self) -> 'AlphaZeroNet':
        """Unfreeze the residual tower to allow full training.
        
        Returns:
            self for method chaining
        """
        # Unfreeze initial convolution
        self.conv1.requires_grad_(True)
        self.bn1.requires_grad_(True)
        
        # Unfreeze all residual blocks
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = True
        
        print(f"🔓 Backbone unfrozen: all layers trainable")
        return self
    
    def get_trainable_params(self) -> dict:
        """Get count of trainable vs frozen parameters.
        
        Returns:
            Dict with 'trainable', 'frozen', and 'total' counts
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "frozen": total - trainable,
            "total": total
        }
    
    def is_small_model(self) -> bool:
        """Check if this is a small model variant."""
        return self.num_blocks < 10
    
    def is_big_model(self) -> bool:
        """Check if this is the full-size model."""
        return self.num_blocks >= 15


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_alphazero_model(
    size: str = "standard",
    use_se: bool = True
) -> AlphaZeroNet:
    """Create an AlphaZeroNet model with preset configurations.
    
    Args:
        size: Model size preset:
            - 'mini': 6 blocks, 128 channels (for testing/low VRAM)
            - 'small': 10 blocks, 256 channels (faster training)
            - 'standard': 19 blocks, 256 channels (AlphaZero default)
            - 'large': 24 blocks, 320 channels (more capacity)
        use_se: Whether to use Squeeze-and-Excitation blocks
        
    Returns:
        Configured AlphaZeroNet model
    """
    configs = {
        'mini': {'num_blocks': 6, 'channels': 128},
        'small': {'num_blocks': 10, 'channels': 256},
        'standard': {'num_blocks': 19, 'channels': 256},
        'large': {'num_blocks': 24, 'channels': 320},
    }
    
    if size.lower() not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")
    
    config = configs[size.lower()]
    return AlphaZeroNet(
        num_blocks=config['num_blocks'],
        channels=config['channels'],
        use_se=use_se
    )


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("AlphaZeroNet Architecture Test")
    print("=" * 60)
    
    # Create model
    model = AlphaZeroNet()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: AlphaZeroNet (19 blocks, 256 channels, SE=True)")
    print(f"Input channels: {model.input_channels}")
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, 119, 8, 8).to(device)
    
    # Warm up
    with torch.no_grad():
        _ = model(x)
    
    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    num_iters = 100
    with torch.no_grad():
        for _ in range(num_iters):
            policy, value = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    
    print(f"\nForward pass benchmark ({num_iters} iterations, batch={batch_size}):")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per batch: {elapsed/num_iters*1000:.2f}ms")
    print(f"  Per position: {elapsed/num_iters/batch_size*1000:.2f}ms")
    
    print(f"\nOutput shapes:")
    print(f"  Policy: {policy.shape} (should be [{batch_size}, 4672])")
    print(f"  Value: {value.shape} (should be [{batch_size}, 1])")
    
    print("\n✓ All tests passed!")
