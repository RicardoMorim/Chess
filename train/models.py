"""
Chess Neural Network Architectures
===================================

Contains:
- ChessNet: Standard ResNet with SE blocks (18 or 22 input channels)
- ESTNet: Early Split Trunk variant (18 input channels)

Factory functions:
- create_model(variant): New unified factory 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = x.view(b, c, -1).mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with SE attention."""
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        x = self.se(x)
        return x + residual


# =============================================================================
# CHESSNET: Standard Architecture
# =============================================================================

class ChessNet(nn.Module):
    """
    Standard AlphaZero-style ResNet for chess.
    
    Supports:
    - 18 input channels (baseline: pieces + game state)
    - 22 input channels (with attack maps)
    """
    def __init__(
        self,
        input_channels,
        num_blocks=15,
        channels=256,
    ):
        super().__init__()
        self.input_channels = input_channels

        self.conv_in = nn.Conv2d(input_channels, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)

        self.blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Policy head (73 planes for move encoding)
        self.policy_conv = nn.Conv2d(channels, 73, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(73)

        # Value head
        self.value_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))

        for block in self.blocks:
            x = block(x)

        # Policy
        policy = self.policy_bn(self.policy_conv(x))
        policy = policy.view(x.size(0), -1)

        # Value
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.mean(dim=[2, 3])
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


# =============================================================================
# ESTNET: Early Split Trunk (Experimental)
# =============================================================================

class ESTNet(nn.Module):
    """
    Early Split Trunk Network.
    
    Separates policy and value processing earlier to reduce gradient interference.
    Total blocks = shared + policy + value = 15 (same capacity as ChessNet).
    """
    def __init__(
        self,
        input_channels=18,
        channels=256,
        shared_blocks=5,
        policy_blocks=5,
        value_blocks=5,
    ):
        super().__init__()
        self.input_channels = input_channels

        assert shared_blocks + policy_blocks + value_blocks == 15, \
            "Total blocks must equal 15 for fair comparison"

        # Input
        self.conv_in = nn.Conv2d(input_channels, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)

        # Shared trunk
        self.shared = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(shared_blocks)]
        )

        # Policy-specific trunk
        self.policy_trunk = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(policy_blocks)]
        )

        # Value-specific trunk
        self.value_trunk = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(value_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 73, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(73)

        # Value head
        self.value_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))

        # Shared processing
        for block in self.shared:
            x = block(x)

        # Branch into separate trunks
        policy_x = x
        value_x = x

        for block in self.policy_trunk:
            policy_x = block(policy_x)

        for block in self.value_trunk:
            value_x = block(value_x)

        # Policy
        policy = self.policy_bn(self.policy_conv(policy_x))
        policy = policy.view(policy.size(0), -1)

        # Value
        value = F.relu(self.value_bn(self.value_conv(value_x)))
        value = value.mean(dim=[2, 3])
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_model(variant: str, **kwargs) -> nn.Module:
    """
    Create a chess model.
    
    Args:
        variant: One of:
            - "baseline": ChessNet with 18 input channels
            - "attack": ChessNet with 22 input channels (attack maps)
            - "est": ESTNet with 18 input channels (early split trunk)
        **kwargs: Additional arguments passed to the model constructor
    
    Returns:
        Configured model instance
    """
    factories = {
        "baseline": lambda: ChessNet(input_channels=18, **kwargs),
        "attack": lambda: ChessNet(input_channels=22, **kwargs),
        "est": lambda: ESTNet(input_channels=18, **kwargs),
    }
    
    if variant not in factories:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(factories.keys())}")
    
    return factories[variant]()



def load_model_with_compatibility(model: nn.Module, checkpoint_path: str, device='cpu') -> nn.Module:
    """
    Load a checkpoint with backward compatibility handling.
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
    
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Try strict loading first, fall back to non-strict
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
        print("Warning: Loaded checkpoint with non-strict matching")
    
    return model
