import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# SQUEEZE-AND-EXCITATION BLOCK (Used by Leela Chess Zero)
# ============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.
    
    This significantly improves chess network performance by allowing
    the network to weight channel importance dynamically.
    """
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        batch, channels, _, _ = x.size()
        # Global average pooling
        y = x.view(batch, channels, -1).mean(dim=2)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        # Scale
        return x * y.view(batch, channels, 1, 1)


# ============================================================================
# RESIDUAL BLOCK WITH SE (Pre-activation style like ResNet v2)
# ============================================================================
class ResidualBlock(nn.Module):
    """Residual block with optional Squeeze-and-Excitation.
    
    Uses pre-activation (BN->ReLU->Conv) pattern which trains better
    for very deep networks.
    """
    def __init__(self, channels, use_se=True, se_reduction=4):
        super(ResidualBlock, self).__init__()
        self.use_se = use_se
        
        # Pre-activation residual block
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        # Optional SE block
        if use_se:
            self.se = SEBlock(channels, se_reduction)

    def forward(self, x):
        residual = x
        
        # Pre-activation pattern: BN -> ReLU -> Conv
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        
        # Apply SE if enabled
        if self.use_se:
            out = self.se(out)
        
        # Residual connection
        out = out + residual
        return out


# ============================================================================
# LEGACY RESIDUAL BLOCK (For backward compatibility with old models)
# ============================================================================
class LegacyResidualBlock(nn.Module):
    """Original residual block for backward compatibility."""
    def __init__(self, channels):
        super(LegacyResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


# ============================================================================
# IMPROVED CHESS NEURAL NETWORK
# ============================================================================
class ChessNet(nn.Module):
    """Improved Chess Neural Network with SE blocks and better architecture.
    
    Key improvements over original:
    - Squeeze-and-Excitation blocks for channel attention
    - Pre-activation residual blocks
    - Global average pooling in value head
    - Larger value head for better position evaluation
    - More residual blocks for the big model (15 instead of 10)
    
    Architecture:
    - Input: Board representation (18 or 22 channels)
    - Body: Residual tower with SE blocks
    - Policy head: Conv -> BN -> Flatten (4672 outputs for move encoding)
    - Value head: Conv -> Global Pool -> FC -> tanh (scalar output)
    """
    def __init__(self, num_blocks=15, channels=256, input_channels=22, use_se=True, legacy_mode=False):
        super(ChessNet, self).__init__()
        self.input_channels = input_channels
        self.num_blocks = num_blocks
        self.channels = channels
        self.use_se = use_se
        self.legacy_mode = legacy_mode
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Residual tower
        if legacy_mode:
            self.blocks = nn.ModuleList([LegacyResidualBlock(channels) for _ in range(num_blocks)])
        else:
            self.blocks = nn.ModuleList([
                ResidualBlock(channels, use_se=use_se, se_reduction=4) 
                for _ in range(num_blocks)
            ])
        
        # Policy head - outputs 73 planes (AlphaZero-style move encoding)
        self.policy_conv = nn.Conv2d(channels, 73, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(73)
        
        # Improved value head with global average pooling
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)  # More channels
        self.value_bn = nn.BatchNorm2d(32)
        # Global average pooling reduces 32x8x8 -> 32
        self.value_fc1 = nn.Linear(32, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual tower
        for block in self.blocks:
            x = block(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = policy.view(-1, 73 * 8 * 8)  # 4672 outputs
        
        # Value head with global average pooling
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.mean(dim=[2, 3])  # Global average pooling: (B, 32, 8, 8) -> (B, 32)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def is_small_model(self):
        """Check if this is the small model architecture"""
        return self.input_channels == 18 and self.num_blocks <= 6
    
    def is_big_model(self):
        """Check if this is the big model architecture"""
        return self.input_channels >= 20 and self.num_blocks >= 10


# ============================================================================
# LEGACY CHESS NETWORK (For loading old models)
# ============================================================================
class LegacyChessNet(nn.Module):
    """Legacy ChessNet for backward compatibility with existing trained models."""
    def __init__(self, num_blocks=10, channels=256, input_channels=20):
        super(LegacyChessNet, self).__init__()
        self.input_channels = input_channels
        self.num_blocks = num_blocks
        
        self.conv1 = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([LegacyResidualBlock(channels) for _ in range(num_blocks)])
        self.policy_conv = nn.Conv2d(channels, 73, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(73)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            x = block(x)
        policy = self.policy_bn(self.policy_conv(x))  
        policy = policy.view(-1, 73 * 8 * 8)
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 64)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value
    
    def is_small_model(self):
        return self.input_channels == 18 and self.num_blocks == 5
    
    def is_big_model(self):
        return self.input_channels == 20 and self.num_blocks == 10


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================
def create_chess_model(model_type="big", use_se=True, legacy=False):
    """Create a chess model based on the specified type.
    
    Args:
        model_type: 'big', 'small', or 'medium'
        use_se: Whether to use Squeeze-and-Excitation blocks (recommended)
        legacy: If True, create legacy model for loading old checkpoints
    
    Model configurations:
    - small: 6 blocks, 18 input channels (fast, for testing)
    - medium: 10 blocks, 22 input channels (balanced)
    - big: 15 blocks, 22 input channels (best quality)
    """
    if legacy:
        # Return legacy model for loading old checkpoints
        if model_type.lower() == "small":
            return LegacyChessNet(num_blocks=5, channels=256, input_channels=18)
        else:
            return LegacyChessNet(num_blocks=10, channels=256, input_channels=20)
    
    # New improved models
    if model_type.lower() == "small":
        return ChessNet(num_blocks=6, channels=256, input_channels=18, use_se=use_se)
    elif model_type.lower() == "medium":
        return ChessNet(num_blocks=10, channels=256, input_channels=22, use_se=use_se)
    else:  # big
        return ChessNet(num_blocks=15, channels=256, input_channels=22, use_se=use_se)


def load_model_with_compatibility(model_path, device='cuda', prefer_new=True):
    """Load a model with automatic architecture detection.
    
    This function attempts to load a saved model and automatically
    determines whether it's a legacy or new architecture.
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model to
        prefer_new: If True, try new architecture first
    
    Returns:
        Loaded model on the specified device
    """
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to detect architecture from checkpoint
    # Check for SE block weights (new architecture)
    has_se = any('se.' in key for key in checkpoint.keys())
    
    # Check input channels from first conv layer
    first_conv_weight = checkpoint.get('conv1.weight', None)
    if first_conv_weight is not None:
        input_channels = first_conv_weight.shape[1]
    else:
        input_channels = 20  # Default
    
    # Count number of blocks
    num_blocks = len([k for k in checkpoint.keys() if k.startswith('blocks.') and k.endswith('.conv1.weight')])
    
    # Determine model type
    if has_se or input_channels == 22:
        # New architecture
        if num_blocks <= 6:
            model = create_chess_model("small", use_se=has_se, legacy=False)
        elif num_blocks <= 10:
            model = create_chess_model("medium", use_se=has_se, legacy=False)
        else:
            model = create_chess_model("big", use_se=has_se, legacy=False)
    else:
        # Legacy architecture
        if input_channels == 18 and num_blocks <= 5:
            model = create_chess_model("small", legacy=True)
        else:
            model = create_chess_model("big", legacy=True)
    
    model.load_state_dict(checkpoint)
    return model.to(device)
