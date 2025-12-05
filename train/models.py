import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
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

# Chess Neural Network with adjustable blocks and channels
class ChessNet(nn.Module):
    def __init__(self, num_blocks=10, channels=256, input_channels=20):
        super(ChessNet, self).__init__()
        self.input_channels = input_channels  # Store this for model identification
        self.num_blocks = num_blocks  # Store this for model identification
        if num_blocks == 10:
            self.conv1 = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])
            self.policy_conv = nn.Conv2d(channels, 73, kernel_size=1)
            self.policy_bn = nn.BatchNorm2d(73)
            self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_fc1 = nn.Linear(64, 256)
            self.value_fc2 = nn.Linear(256, 1)
        else:
            self.conv1 = nn.Conv2d(18, channels, kernel_size=3, padding=1)  # 18 channels
            self.bn1 = nn.BatchNorm2d(channels)
            self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])
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
        """Check if this is the small model architecture"""
        return self.input_channels == 18 and self.num_blocks == 5
    
    def is_big_model(self):
        """Check if this is the big model architecture"""
        return self.input_channels == 20 and self.num_blocks == 10

# Factory function to create models based on type
def create_chess_model(model_type="big"):
    """Create a chess model based on the specified type
    
    Args:
        model_type: Either 'big' (20 channels, 10 blocks) or 'small' (18 channels, 5 blocks)
    """
    if model_type.lower() == "small":
        return ChessNet(num_blocks=5, channels=256, input_channels=18)
    else:  # Default to big model
        return ChessNet(num_blocks=10, channels=256, input_channels=20)
