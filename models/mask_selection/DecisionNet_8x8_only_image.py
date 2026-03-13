"""
DecisionNet for 8x8 mask selection using direct compressed measurements.

This model takes compressed measurement (1 channel) as input
instead of reconstructed video, eliminating redundant reconstruction overhead.
"""

import torch
import torch.nn as nn
import random
from collections import deque


class DecisionNet(nn.Module):
    def __init__(self, num_classes=16):
        super(DecisionNet, self).__init__()
        # Input: compressed measurement (1 channel) = 1 channels total
        # REMOVED InstanceNorm to test if it was removing critical input information
        # self.input_norm = nn.InstanceNorm2d(17)

        # Note: Using LayerNorm instead of BatchNorm to support batch_size=1 during training
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, 64),  # GroupNorm: works with any batch size
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, 128),  # GroupNorm: works with any batch size
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, 256),  # GroupNorm: works with any batch size
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),  # LayerNorm: works with any batch size
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
            # Removed: nn.Softmax(dim=1) - apply softmax explicitly when needed
        )

        # Improve weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, compressed_measurement, return_logits=False):
        """
        Args:
            compressed_measurement: [B, 1, 8, 8] - compressed video measurement
            mask: [B, 16, 8, 8] - binary sampling mask used for compression
            return_logits: if True, return raw logits; if False, return probabilities
        Returns:
            [B, num_classes] - logits or probability distribution over next mask selection
        """
        # Concatenate compressed measurement and mask
        x = torch.cat([compressed_measurement], dim=1)  # [B, 17, 8, 8]
        # REMOVED: x = self.input_norm(x) - testing without input normalization
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        if return_logits:
            return logits
        else:
            return torch.nn.functional.softmax(logits, dim=1)

    def get_logits(self, compressed_measurement):
        """Get raw logits (for training with proper loss functions)"""
        return self.forward(compressed_measurement, return_logits=True)

    def get_probabilities(self, compressed_measurement):
        """Get probability distribution (for inference/evaluation)"""
        return self.forward(compressed_measurement, return_logits=False)

class DecisionNetWithEpsilonGreedy(DecisionNet):
    def __init__(self, num_classes=16, epsilon_start=1.0, epsilon_end=0.001, epsilon_decay=0.995):
        super().__init__(num_classes)
        self.num_classes = num_classes
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def forward_with_exploration(self, compressed_measurement, training=True):
        """
        Forward pass with epsilon-soft exploration strategy.

        Instead of choosing between 100% model or 100% random (one-hot),
        we blend model predictions with uniform distribution based on epsilon.
        This encourages diversity while still using learned preferences.
        """
        device = compressed_measurement.device
        batch_size = compressed_measurement.size(0)

        # Get model predictions (probabilities)
        model_probs = super().forward(compressed_measurement, return_logits=False)

        if not training:
            # During evaluation, use pure model predictions
            return model_probs

        # Epsilon-soft strategy: blend model predictions with uniform distribution
        # epsilon = 0: pure model predictions (exploitation)
        # epsilon = 1: uniform distribution (maximum exploration)
        uniform_probs = torch.ones_like(model_probs) / self.num_classes

        # Blend: (1 - epsilon) * model + epsilon * uniform
        blended_probs = (1 - self.epsilon) * model_probs + self.epsilon * uniform_probs

        return blended_probs

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.epsilon

class ExperienceBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def append(self, experience):
        """Add experience"""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample batch_size experiences randomly from buffer"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

class DecisionNetWithExpReplay(DecisionNetWithEpsilonGreedy):
    def __init__(self, num_classes=16, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, buffer_size=30000):
        super().__init__(num_classes, epsilon_start, epsilon_end, epsilon_decay)
        self.experience_buffer = ExperienceBuffer(buffer_size)