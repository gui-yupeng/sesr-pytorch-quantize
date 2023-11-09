import torch.nn as nn
import torch
import torch.nn.functional as F

class CollapsibleLinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size, activation='prelu'):
        super().__init__()

        self.conv_expand = nn.Conv2d(in_channels, tmp_channels, (kernel_size, kernel_size),
                                     padding=int((kernel_size - 1) / 2), bias=False)
        self.conv_squeeze = nn.Conv2d(tmp_channels, out_channels, (1, 1))

        if activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'identity':
            self.activation = nn.Identity()
        else:
            raise Exception(f'Activation not supported: {activation}')

        self.collapsed=False

    def forward(self, x):
        if self.collapsed:
            return self.activation(self.conv_expand(x))
        return self.activation(self.conv_squeeze(self.conv_expand(x)))

    def collapse(self):
        if self.collapsed:
            print('Already collapsed')
            return

        padding = int((self.conv_expand.kernel_size[0] - 1)/ 2)
        new_conv = nn.Conv2d(self.conv_expand.in_channels,
                             self.conv_squeeze.out_channels,
                             self.conv_expand.kernel_size,
                             padding=padding)

        # Find corresponding kernel weights by applying the convolutional block
        # to a delta function (center=1, 0 everywhere else)
        delta = torch.eye(self.conv_expand.in_channels)
        delta = delta.unsqueeze(2).unsqueeze(3)
        k = self.conv_expand.kernel_size[0]
        pad = int((k - 1) / 2)  # note: this will probably break if k is even
        delta = F.pad(delta, (pad, pad, pad, pad))  # Shape: in_channels x in_channels x kernel_size x kernel_size
        delta = delta.to(self.conv_expand.weight.device)

        with torch.no_grad():
            bias = self.conv_squeeze.bias
            kernel_biased = self.conv_squeeze(self.conv_expand(delta))
            kernel = kernel_biased - bias[None, :, None, None]

        # Flip and permute
        kernel = torch.flip(kernel, [2, 3])
        kernel = kernel.permute([1, 0, 2, 3])

        # Assign weight and return
        new_conv.weight = nn.Parameter(kernel)
        new_conv.bias = bias

        # Replace current layers
        self.conv_expand = new_conv
        self.conv_squeeze = nn.Identity()

        self.collapsed = True


class ResidualCollapsibleLinearBlock(CollapsibleLinearBlock):
    """
    Residual version of CollapsibleLinearBlock.
    """

    def forward(self, x):
        if self.collapsed:
            return self.activation(self.conv_expand(x))
        return self.activation(x + self.conv_squeeze(self.conv_expand(x)))

    def collapse(self):
        if self.collapsed:
            print('Already collapsed')
            return
        super().collapse()
        middle = self.conv_expand.kernel_size[0] // 2
        num_channels = self.conv_expand.in_channels
        with torch.no_grad():
            for idx in range(num_channels):
                self.conv_expand.weight[idx, idx, middle, middle] += 1.

class AddOp(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2
    