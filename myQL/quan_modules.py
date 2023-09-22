import torch
from torch import nn


class myMinMaxObserver(nn.Module):
    """To record the max/min value as register in network"""
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))