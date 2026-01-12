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
    - Dropout for regularization (prevents overfitting)
    
    Architecture:
    - Input: Board representation (18 or 22 channels)
    - Body: Residual tower with SE blocks
    - Policy head: Conv -> BN -> Dropout -> Flatten (4672 outputs for move encoding)
    - Value head: Conv -> Global Pool -> FC -> Dropout -> FC -> tanh (scalar output)
    """
    def __init__(self, num_blocks=15, channels=256, input_channels=22, use_se=True, 
                 legacy_mode=False, policy_dropout=0.1, value_dropout=0.3):
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
        self.policy_dropout = nn.Dropout2d(p=policy_dropout)  # Spatial dropout for conv layers
        
        # Improved value head with global average pooling
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)  # More channels
        self.value_bn = nn.BatchNorm2d(32)
        # Global average pooling reduces 32x8x8 -> 32
        self.value_fc1 = nn.Linear(32, 128)
        self.value_dropout = nn.Dropout(p=value_dropout)  # Dropout after FC layer
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual tower
        for block in self.blocks:
            x = block(x)
        
        # Policy head with dropout
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_dropout(policy)  # Apply spatial dropout
        policy = policy.view(-1, 73 * 8 * 8)  # 4672 outputs
        
        # Value head with global average pooling and dropout
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.mean(dim=[2, 3])  # Global average pooling: (B, 32, 8, 8) -> (B, 32)
        value = F.relu(self.value_fc1(value))
        value = self.value_dropout(value)  # Apply dropout
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def freeze_backbone(self):
        """Freeze the residual tower for fine-tuning only the heads.
        
        This is useful when training on a small dataset (like puzzles) to prevent
        overfitting. The pretrained backbone features are preserved while only
        the policy and value heads are updated.
        """
        # Freeze initial convolution
        self.conv1.requires_grad_(False)
        self.bn1.requires_grad_(False)
        
        # Freeze all residual blocks
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        print(f"🔒 Backbone frozen: {self.num_blocks} residual blocks + initial conv")
        return self
    
    def unfreeze_backbone(self):
        """Unfreeze the residual tower to allow full training."""
        # Unfreeze initial convolution
        self.conv1.requires_grad_(True)
        self.bn1.requires_grad_(True)
        
        # Unfreeze all residual blocks
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = True
        
        print(f"🔓 Backbone unfrozen: all layers trainable")
        return self
    
    def get_trainable_params(self):
        """Get count of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable
        return {"trainable": trainable, "frozen": frozen, "total": total}
    
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
# LIMITED MODEL (Optimized for low VRAM - 2GB GPUs like GTX 1050)
# ============================================================================
class LimitedChessNet(nn.Module):
    """Memory-efficient Chess Neural Network for low VRAM GPUs (2GB).
    
    Optimizations:
    - Only 4 residual blocks (vs 10-15)
    - 64 filters (vs 256) - 16x fewer parameters
    - No SE blocks (saves memory)
    - 18 input channels (no attack maps)
    - Smaller value head
    
    This model is designed to fit in 2GB VRAM with batch size 16.
    Estimated VRAM usage: ~800MB model + ~600MB gradients + ~400MB batch = ~1.8GB
    """
    def __init__(self, num_blocks=4, channels=64, input_channels=18):
        super(LimitedChessNet, self).__init__()
        self.input_channels = input_channels
        self.num_blocks = num_blocks
        self.channels = channels
        self.use_se = False
        self.legacy_mode = False
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Residual tower (lightweight blocks)
        self.blocks = nn.ModuleList([
            self._make_block(channels) for _ in range(num_blocks)
        ])
        
        # Policy head - 73 planes for move encoding
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)  # Smaller intermediate
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 64, 4672)  # 4672 = 73 * 64 move outputs
        
        # Compact value head with global pooling
        self.value_conv = nn.Conv2d(channels, 16, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_block(self, channels):
        """Create a simple residual block without SE."""
        return nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )
    
    def _initialize_weights(self):
        """Initialize weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual tower with skip connections
        for block in self.blocks:
            residual = x
            x = block(x) + residual
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # Value head with global average pooling
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.mean(dim=[2, 3])  # Global avg pool: (B, 16, 8, 8) -> (B, 16)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def is_small_model(self):
        return True
    
    def is_big_model(self):
        return False


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================
def create_chess_model(model_type="big", use_se=True, legacy=False):
    """Create a chess model based on the specified type.
    
    Args:
        model_type: 'big', 'small', 'medium', or 'limited'
        use_se: Whether to use Squeeze-and-Excitation blocks (recommended)
        legacy: If True, create legacy model for loading old checkpoints
    
    Model configurations:
    - limited: 4 blocks, 64 channels, 18 inputs (for 2GB VRAM GPUs)
    - small: 6 blocks, 256 channels, 18 inputs (fast, for testing)
    - medium: 10 blocks, 256 channels, 20 inputs (balanced)
    - big: 15 blocks, 256 channels, 22 inputs (best quality)
    """
    if model_type.lower() == "limited":
        # Special memory-efficient model for low VRAM GPUs
        return LimitedChessNet(num_blocks=4, channels=64, input_channels=18)
    
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
        return ChessNet(num_blocks=10, channels=256, input_channels=20, use_se=use_se)
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
        num_filters = first_conv_weight.shape[0]
    else:
        input_channels = 20  # Default
        num_filters = 256
    
    # Count number of blocks
    num_blocks = len([k for k in checkpoint.keys() if k.startswith('blocks.') and k.endswith('.0.weight')])
    if num_blocks == 0:
        num_blocks = len([k for k in checkpoint.keys() if k.startswith('blocks.') and k.endswith('.conv1.weight')])
    
    # Check for limited model (64 filters, 4 blocks, has policy_fc)
    is_limited = num_filters <= 64 or 'policy_fc.weight' in checkpoint.keys()
    
    # Determine model type
    if is_limited:
        model = create_chess_model("limited")
    elif has_se or input_channels == 22:
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
