import torch
import torch.nn as nn
import torch.nn.functional as F

class WarpKeyframe(nn.Module):
    def __init__(self, height, width, clip_size, device=None):
        super().__init__()
        self.flow_grid = torch.stack(torch.meshgrid(torch.arange(0, height), torch.arange(0, width)), -1).float() #[H, W, 2]
        self.flow_grid = torch.flip(self.flow_grid, (-1,)) # from (y, x) to (x, y)
        self.flow_grid = self.flow_grid.unsqueeze(0) #[H, W, 2] -> [1, H, W, 2]
        self.flow_grid = self.flow_grid.cuda()

        self.height = height
        self.width = width
        self.clip_size = clip_size
        
    def extra_repr(self):
        return 'height={}, width={}, clip_size={}'.format(self.height, self.width, self.clip_size)

    def forward(self, key_frame, output_flow):
        B, C, T, H, W = output_flow.shape
        output_flow = output_flow.permute(0, 2, 3, 4, 1).contiguous().view(B*T, H, W, C) #[B, 2, T, H, W] -> [BT, H, W, 2]
        key_frame = key_frame.permute(0, 2, 1, 3, 4).expand(-1, T, -1, -1, -1).contiguous().view(B*T, -1, H, W) #[B, C, 1, H, W] -> [B, 1, C, H, W] -> [BT, C, H, W]

        next_coords = self.flow_grid.to(output_flow) + output_flow
        next_coords = 2 * next_coords / torch.tensor([[[[W-1, H-1]]]]).to(next_coords) - 1 

        image_warp = F.grid_sample(key_frame, next_coords, padding_mode='border', align_corners=True)

        image_warp = image_warp.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4) # [BT, C, H, W] -> [B, C, T, H, W]
        return image_warp
