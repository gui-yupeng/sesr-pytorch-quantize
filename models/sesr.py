import torch
from torch import nn
from models.model_utils_pt import *

class sesr(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 num_channels=16,
                 num_lblocks=3,
                 scaling_factor=4):
        super().__init__()

        self.conv_first = CollapsibleLinearBlock(in_channels=in_channels, out_channels=num_channels,
                                                        tmp_channels=256, kernel_size=5, activation='relu')

        residual_layers = [
            ResidualCollapsibleLinearBlock(in_channels=num_channels, out_channels=num_channels,
                                                  tmp_channels=256, kernel_size=3, activation='relu')
            for _ in range(num_lblocks)
        ]
        self.residual_block = nn.Sequential(*residual_layers)

        self.add_residual = AddOp()

        self.conv_last = CollapsibleLinearBlock(in_channels=num_channels,
                                                       out_channels=out_channels * scaling_factor ** 2,
                                                       tmp_channels=256, kernel_size=5, activation='identity')

        #self.add_upsampled_input = AddOp()
        self.depth_to_space = nn.PixelShuffle(scaling_factor)

    def collapse(self):
        self.conv_first.collapse()
        for layer in self.residual_block:
            layer.collapse()
        self.conv_last.collapse()

    def before_quantization(self):
        self.collapse()

    def forward(self, input):
        initial_features = self.conv_first(input)  # Extract features from conv-first
        residual_features = self.residual_block(initial_features)  # Get residual features with `lblocks`
        residual_features = self.add_residual(residual_features, initial_features)  # Add init_features and residual
        # print(torch.max(residual_features))
        output = self.conv_last(residual_features)  # Get final features from conv-last
        # output = self.add_upsampled_input(final_features, upsampled_input)  # Add final_features and upsampled_input
        # print(torch.max(output))

        return self.depth_to_space(output)  # Depth-to-space and return