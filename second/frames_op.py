from IPython import embed

from second.core import region_similarity
import numpy as np
import draw_tools
import torch

def handle_frames(previous_frame, current_frame):
    # prev_img_name = previous_frame['metadata'][0]['image_idx']
    # curr_img_name = current_frame['metadata'][0]['image_idx']

    rs = region_similarity.RotateIouSimilarity()
    prev_gt_boxes = previous_frame['gt_boxes']
    curr_gt_boxes = current_frame['gt_boxes']
    # [N, 5] [x,y,w,l,r]
    iou_list = rs.compare(curr_gt_boxes[:, (0,1,3,4,6)], prev_gt_boxes[:, (0,1,3,4,6)]) # shape: [N,M]

    # Parameters
    iou_threshold = 0.33 # if iou >= threshold, we assume same car appears in both frames 
    w_size = 200
    h_size = 176
    # box_per_loc = 2 # it should always be 2

    corr_size = 9
    corr_ratio = corr_size//2 
    '''
    Generate offset / Masks
    Masks: We do warp on rois to current frame
    '''
    current_idx, previous_idx = np.where(iou_list >= iou_threshold)
    previous_list = prev_gt_boxes[previous_idx]
    current_list = curr_gt_boxes[current_idx]
    num_of_boxes = current_list.shape[0]
    gt_offset = np.zeros([num_of_boxes,2])
    
    # Masks
    # Masks gt boxes in current frames (Rotations are considered)
    # sqrt_wl = np.sqrt(current_list[:,3]*current_list[:,3] + current_list[:,4]*current_list[:,4])/2
    # car_theta = np.arctan(current_list[:,3]/current_list[:,4]) # origin
    cls_loc_in_tensor = current_list[:,0:2] #xy
    # lw_in_tensor = np.flip(current_list[:,3:5],1) #lw
    lw_in_tensor = current_list[:,3:5]
    unrotated_right_top =  lw_in_tensor/2
    unrotated_left_bottom = -lw_in_tensor/2
    unrotated_left_top = lw_in_tensor/2
    unrotated_right_bottom = -lw_in_tensor/2
    unrotated_left_top[:,0]=  unrotated_left_top[:,0] - lw_in_tensor[:,0]
    unrotated_right_bottom[:,0] =  unrotated_right_bottom[:,0] + lw_in_tensor[:,0]

    theta = current_list[:,6] 
    # sum_theta = theta + car_theta #rotated
    
    rot_sin = np.sin(theta)
    rot_cos = np.cos(theta)
    rot_mat_T = np.array(
        [[rot_cos, -rot_sin], [rot_sin, rot_cos]], dtype=curr_gt_boxes.dtype)

    rotated_coord = np.ndarray([num_of_boxes,4,2]) #[num,(LB,RT,RB,LT),(x,y)]
    
    for i in range(num_of_boxes):
        rotated_coord[i, 0, :] = unrotated_left_bottom[i, :] @ rot_mat_T[..., i]
        rotated_coord[i, 1, :] = unrotated_right_top[i, :] @ rot_mat_T[..., i]
        rotated_coord[i, 2, :] = unrotated_right_bottom[i, :] @ rot_mat_T[..., i]
        rotated_coord[i, 3, :] = unrotated_left_top[i, :] @ rot_mat_T[..., i]

    rotated_coord[:, 0, :] = ((cls_loc_in_tensor + rotated_coord[:, 0, :]) * 2.5).round() 
    rotated_coord[:, 1, :] = ((cls_loc_in_tensor + rotated_coord[:, 1, :]) * 2.5).round()
    rotated_coord[:, 2, :] = ((cls_loc_in_tensor + rotated_coord[:, 2, :]) * 2.5).round() 
    rotated_coord[:, 3, :] = ((cls_loc_in_tensor + rotated_coord[:, 3, :]) * 2.5).round()

    mask_coordinates = np.ndarray([num_of_boxes,4],dtype=int) #[num, (x_min,y_min,x_max,y_max)], shape [num,(x,y)]

    # Where we need masks        
    mask_coordinates[:, 0] = np.amin(rotated_coord[..., 0],axis=1)
    mask_coordinates[:, 1] = np.amin(rotated_coord[..., 1],axis=1)
    mask_coordinates[:, 2] = np.amax(rotated_coord[..., 0],axis=1)                
    mask_coordinates[:, 3] = np.amax(rotated_coord[..., 1],axis=1) 
    
    # Get gt offsets
    offset_x_list = ((-((current_list[:,0] -previous_list[:,0]) * 2.5).round() + corr_ratio ) * 2 + 1) / corr_size - 1
    offset_y_list = ((((current_list[:,1] -previous_list[:,1]) * 2.5).round() + corr_ratio ) * 2 + 1) / corr_size - 1

    gt_offset[..., 0] = offset_x_list
    gt_offset[..., 1] = offset_y_list
    # Merge batches
    gt_offset_padded = np.zeros([1,250,2])
    offset_coords_padded = np.zeros([1,250,4])
    target_num = gt_offset.shape[0]

    offset_masks = np.zeros([w_size,h_size,2],dtype=int)
    offset_targets = -np.ones([w_size,h_size,2],
                    dtype=current_frame['reg_targets'].dtype)
    # Coordinate shifts
    mask_coordinates[:,1] = mask_coordinates[:,1] + 100
    mask_coordinates[:,3] = mask_coordinates[:,3] + 100


    for k in range(num_of_boxes):
        offset_targets[mask_coordinates[k,1]:mask_coordinates[k,3],mask_coordinates[k,0]:mask_coordinates[k,2],:] = gt_offset[k]
        offset_masks[mask_coordinates[k,1]:mask_coordinates[k,3],mask_coordinates[k,0]:mask_coordinates[k,2],:] = 1
    
    # If there is no targets, set offset as -1
    offset_targets = offset_targets.reshape(-1,2)
    offset_masks = offset_masks.reshape(-1,2)
    
    gt_offset_padded[:, :gt_offset.shape[0], :] = gt_offset
    offset_coords_padded[:,:mask_coordinates.shape[0],:] = mask_coordinates
    
    current_frame.update({'box_offsets': gt_offset_padded,
                        'offset_coords': offset_coords_padded,
                        'offset_targets': offset_targets,
                        'target_num': target_num,
                        'offset_masks': offset_masks,})

    # '''check offset'''
    # offset_images = offset_masks.reshape(200, 176, 2)

    # offset_images = np.where(offset_images < 0, 0, offset_images)

    # offset_images = offset_images[..., 0] + offset_images[..., 1]

    # import imageio
    # imageio.imwrite("offset_images.png", offset_images)
    # draw_tools.draw_boxes(prev_gt_boxes, 'prev_img')
    # draw_tools.draw_boxes(curr_gt_boxes, 'curr_img')
    return previous_frame, current_frame

def get_response_offset(corr_response, 
                        offset_mask,
                        patch_size, 
                        kernel_size=3, 
                        voting_range=6, 
                        dilation_patch=1):
    n, patch_size, _, h, w= corr_response.shape
    shift = patch_size // 2
    corr_response = corr_response.reshape(n, patch_size * patch_size, h, w)
    offset_mask = offset_mask.reshape(n, h, w, -1).permute(0, 3, 1, 2).contiguous()
    response_sum = torch.zeros([n, patch_size * patch_size, h, w], device=corr_response.device)
    delta_map = torch.zeros([n, 2, h, w],device=corr_response.device)

    for i in range(-voting_range, voting_range + 1):
        for j in range(-voting_range,voting_range + 1):
            response_sum[:, :, max(0, 0+i):min(h, h+i), max(0, 0+j):min(w, w+j)] += \
            corr_response[:, :, max(0, 0-i):min(h, h-i), max(0, 0-j):min(w, w-j)] 

    '''get argmax of channel response'''
    argmax_response_map = response_sum.max(1)[1] # [n, h, w]
    delta_y = argmax_response_map // patch_size - shift
    pos_offset_x = argmax_response_map % patch_size # [n, h, w]
    neg_offset_x = -(argmax_response_map % -patch_size) # [n, h, w]
    delta_x = torch.where(argmax_response_map > 0, pos_offset_x, neg_offset_x)  - shift# [n, h, w]
    delta_map[:,1] = delta_x * dilation_patch
    delta_map[:,0] = delta_y * dilation_patch

    masked_offset_map = delta_map * offset_mask
    # np.savetxt("argmax_response_map.csv", argmax_response_map.reshape(h, w).cpu().numpy())
    # import imageio
    # imageio.imwrite("argmax_response.png", argmax_response_map.reshape(h, w).cpu().numpy())
    return masked_offset_map