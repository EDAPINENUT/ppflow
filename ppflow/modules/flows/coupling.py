import torch
import torch.nn as nn

from .spline import (
    circular_quadratic_spline,
)


class ContextualCircularSplineCouplingLayer(nn.Module):

    def __init__(self, n_dims, mapping_dims, n_context_dims, n_hidden_dims, n_spline_bins):
        super().__init__()
        self.n_dims = n_dims
        self.n_mapping_dims = len(mapping_dims)
        self.n_condition_dims = n_dims - len(mapping_dims)
        self.n_context_dims = n_context_dims
        self.n_spline_bins = n_spline_bins
        self.mapping_dims = mapping_dims
        self.condition_dims = [i for i in range(n_dims) if i not in mapping_dims]

        self.condition_net = nn.Sequential(
            nn.Linear(3*self.n_condition_dims + n_context_dims, n_hidden_dims), nn.LeakyReLU(),
            nn.Linear(n_hidden_dims, n_hidden_dims), nn.LeakyReLU(),
            nn.Linear(n_hidden_dims, self.n_mapping_dims*n_spline_bins*3)
        )

    def _get_spline_params(self, x_cond, c):
        inputs = torch.cat([
            x_cond, torch.sin(x_cond), torch.cos(x_cond), c
        ], dim=-1)
        outs = self.condition_net(inputs)   # (..., n_mapping_dims * nbins * 3)
        shape = list(outs.size())[:-1] + [self.n_mapping_dims, self.n_spline_bins, 3]
        outs = outs.reshape(shape)
        w, h, d = torch.unbind(outs, dim=-1)
        return w, h, d

    def _merge_and_sortdim(self, v_map, v_cond):
        v_unsort = torch.cat([v_map, v_cond], dim=-1)
        idx = [0] * v_unsort.size(-1)
        for i, j in enumerate(self.mapping_dims + self.condition_dims):
            idx[j] = i
        v = v_unsort[..., idx]
        return v

    def forward(self, x, c, inverse):
        """
        Args:
            x:  (..., n_dims)
            c:  (..., n_context_dims)
        """
        x_map, x_cond = x[..., self.mapping_dims], x[..., self.condition_dims]
        w, h, d = self._get_spline_params(x_cond, c)  # (..., n_mapping_dims, n_bins)

        y_map, logabsdet = circular_quadratic_spline(
            inputs = x_map,
            unnormalized_widths = w,
            unnormalized_heights = h,
            unnormalized_derivatives = d,
            inverse = inverse,
        )
        y = self._merge_and_sortdim(v_map=y_map, v_cond=x_cond)
        logabsdet = logabsdet.sum(dim=-1)
        return y, logabsdet

