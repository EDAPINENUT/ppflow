import torch
import torch.nn as nn


class ContextualSequentialFlow(nn.Module):

    def __init__(self, flows: list):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x, c, inverse):
        if inverse:
            sum_logabsdet = 0
            for flow in reversed(self.flows):
                x, logabsdet = flow(x, c, inverse=True)
                sum_logabsdet += logabsdet
            return x, sum_logabsdet
        else:
            sum_logabsdet = 0
            for flow in self.flows:
                x, logabsdet = flow(x, c, inverse=False)
                sum_logabsdet = sum_logabsdet + logabsdet
            return x, sum_logabsdet
