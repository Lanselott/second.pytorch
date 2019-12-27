from frames_op import get_response_offset
import torch
from IPython import embed
from resample2d import Resample2d

device="cuda:0"

sample1 = torch.rand([1, 9, 9, 10, 10], device=device)
sample2 = torch.ones([1, 9, 9, 10, 10], device=device)
previous_map = torch.randint(1000, (1, 1, 10, 10), device=device).float()

offset_mask_1 = torch.zeros([1, 10, 10, 2], device=device) # n, h, w, -1
offset_mask_2 = torch.ones([1, 10, 10, 2], device=device)

offset_mask_1[0, 4:7, 4:7, :] = 1
sample1[0, 2, 2, 5, 5] = 10
# sample2[0, 4, 4] = 1

if __name__ == "__main__":
    resample_layer = Resample2d()
    masked_offset_map = get_response_offset(sample1, offset_mask=offset_mask_1, patch_size=9, kernel_size=3, voting_range=1, dilation_patch=1)
    # offset_map = get_response_offset(sample1, offset_mask=offset_mask_2, patch_size=9, kernel_size=3, voting_range=1, dilation_patch=1)

    warped_map = resample_layer(previous_map, masked_offset_map)
    embed()