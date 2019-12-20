from frames_op import get_response_offset
import torch
from IPython import embed

sample1 = torch.zeros([1, 9, 9, 10, 10])
sample2 = torch.zeros([1, 9, 9, 10, 10])
offset_mask = torch.ones([1, 2, 10, 10])
sample1[0, 3, 4] = 1
# sample2[0, 4, 4] = 1

if __name__ == "__main__":
    masked_offset_map = get_response_offset(sample1, offset_mask=offset_mask, patch_size=9, kernel_size=3, voting_range=1, dilation_patch=1)
    embed()