### code adapted from https://github.com/abhyantrika/mediainr ###

import torch


class NormalizedCoordinate(torch.nn.Module):
    def __init__(self, align_corners=False):
        super().__init__()
        self.align_corners = align_corners

    def extra_repr(self):
        s = 'align_corners={align_corners}'
        return s.format(**self.__dict__)

    def normalize_index(self, x: torch.Tensor, xmax: float, align_corners: bool=True):
        if xmax == 1.:
            return x * 0.
        elif align_corners:
            step = 2. / (xmax - 1)
            return -1. + x * step
        else:
            step = 2. / xmax
            return -1. + step / 2. + x * step

    def forward(self, l: torch.IntTensor, L: int):
        return self.normalize_index(l, float(L), self.align_corners)


def _compute_pixel_idx_nd(n, idx, idx_max, sizes, padding, clipped=True, return_mask=True):
    assert idx.ndim == 2 and idx.shape[1] == n
    assert len(idx_max) == n
    assert all(sizes[d] % idx_max[d] == 0 for d in range(n))
    patch_sizes = [sizes[d] // idx_max[d] for d in range(n)]
    patch_sizes_padded = [patch_sizes[d] + padding[d] * 2 for d in range(n)]
    px_idx = [idx[:, d][:, None] * patch_sizes[d] - padding[d] + torch.arange(patch_sizes_padded[d], device=idx.device)[None, :] for d in range(n)]
    px_idx_clipped = [torch.clip(px_idx[d], 0, sizes[d] - 1) for d in range(n)] if clipped else px_idx
    if return_mask:
        idx_pad_mask = [(px_idx[d] >= 0) * (px_idx[d] < sizes[d]) for d in range(n)]
        return px_idx_clipped, idx_pad_mask
    else:
        return px_idx_clipped


def compute_pixel_idx_1d(idx, idx_max, sizes, padding, clipped=True, return_mask=True):
    """
    Get 1D pixel indexes.
    """
    return _compute_pixel_idx_nd(1, idx, idx_max, sizes, padding, clipped=clipped, return_mask=return_mask)


def compute_pixel_idx_3d(idx, idx_max, sizes, padding, clipped=True, return_mask=True):
    """
    Get 3D pixel indexes.
    """
    return _compute_pixel_idx_nd(3, idx, idx_max, sizes, padding, clipped=clipped, return_mask=return_mask)


def interpolate3D(grid: torch.Tensor, coor_wht: torch.Tensor, align_corners: bool=False):
    """
    Inputs:
        grid: input tensor with shape [T_grid, H_grid, W_grid, C].
        coor: coordinates with shape [N, T, H, W, 3]. In (x, y, t)/(w, h, t) order and normalized in [-1., 1.].
    Output:
        a tensor with shape [N, T, H, W, C]
    """
    _, _, _, C = grid.shape
    N, T, H, W, _ = coor_wht.shape
    # [T_grid, H_grid, W_grid, C] -> [1, C, T_grid, H_grid, W_grid]
    grid = grid.permute(3, 0, 1, 2).unsqueeze(0)
    # [N, T, H, W, 3] -> [1, N * T, H, W, 3]
    coor_wht = coor_wht.view(1, N * T, H, W, 3)
    # 'bilinear' with 5D input is actually 'trilinear' in grid_sample
    return F.grid_sample(grid, coor_wht, mode='bilinear', padding_mode='border', align_corners=align_corners).view(C, N, T, H, W).permute(1, 2, 3, 4, 0)


class GridTrilinear3D(torch.nn.Module):
    """
    The module for mapping feature maps to a fixed size with trilinear interpolation.
    """
    def __init__(self, output_size, align_corners=False):
        super().__init__()
        self.output_size = output_size
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor):
        #x = F.interpolate(x, size=self.output_size, mode='trilinear', align_corners=self.align_corners)
        T, H, W, C = x.shape
        #print(x.shape, self.output_size)
        assert H == self.output_size[1] and W == self.output_size[2], 'F.interpolate has incorrect results in some cases, so use only temporal scale'
        x = x.view(1, 1, T, H * W * C)
        x = F.interpolate(x, size=(self.output_size[0], H * W * C), mode='bilinear', align_corners=self.align_corners)
        x = x.view(self.output_size + (C,))
        return x
