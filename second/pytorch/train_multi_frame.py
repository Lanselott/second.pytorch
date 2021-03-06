import copy
import json
import os
from pathlib import Path
import pickle
import shutil
import time
import re 
import fire
import numpy as np
import torch
from google.protobuf import text_format

import second.data.kitti_common as kitti
import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.core import box_np_ops
from second.data.preprocess import merge_tracking_second_batch, merge_second_batch_multigpu
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_track_builder)
from second.utils.log_tool import SimpleModelLog
from second.utils.progress_bar import ProgressBar
import psutil

from IPython import embed
from frames_op import handle_frames
from collections import defaultdict

def merge_list_inputs(examples, ):
    '''
    ['voxels', 'num_points', 'coordinates', 'num_voxels', 'metrics', 'calib', 'anchors', 'anchors_mask', 'gt_names', 'labels', 'reg_targets', 'gt_boxes', 'importance', 'metadata']
    '''
    example_merged = defaultdict(list)
    for example in examples:
        for k, v in example.items():
            if type(v) == list:
                example_merged[k].append(v[0])
            else:
                example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_gt', 'voxel_labels', 'gt_names', 'gt_classes', 'gt_boxes', 'offset_masks'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'metadata':
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key == 'coordinates':
            # pass # BUG
            coors = []
            for i, coor in enumerate(elems):
                '''
                concat coors to batch coors
                '''
                coor[..., 0] = i
                coors.append(coor)
            ret[key] = np.concatenate(coors, axis=0)
        elif key == 'metrics':
            ret[key] = elems
        else:
            ret[key] = np.stack(elems, axis=0)
        # key == 'anchors' or key == 'anchors_mask' or key == 'labels' or key == 'reg_targets':
        #     ret[key] = elems
    # batch, 1, ... - >  batch, ...
    keys = ret.keys()
    ret['anchors'] = ret['anchors'].squeeze(axis=1)

    if 'anchors_mask' in keys:
        ret['anchors_mask'] = ret['anchors_mask'].squeeze(axis=1)
    if 'labels' in keys:
        ret['labels'] = ret['labels'].squeeze(axis=1)
    if 'reg_targets' in keys:
        ret['reg_targets'] = ret['reg_targets'].squeeze(axis=1)
    if 'importance' in keys:
        ret['importance'] = ret['importance'].squeeze(axis=1)
    if 'offset_coords' in keys:
        ret['offset_coords'] = ret['offset_coords'].squeeze(axis=1)

    return ret
        

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance", "offset_masks"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        else:
            example_torch[k] = v
    return example_torch


def build_network(model_cfg, measure_time=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim
    net = second_track_builder.build(
        model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    return net

def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

def freeze_params(params: dict, include: str=None, exclude: str=None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    remain_params = []
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                continue 
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                continue 
        remain_params.append(p)
    return remain_params

def freeze_params_v2(params: dict, include: str=None, exclude: str=None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                p.requires_grad = False
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                p.requires_grad = False

def filter_param_dict(state_dict: dict, include: str=None, exclude: str=None):
    assert isinstance(state_dict, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    res_dict = {}
    for k, p in state_dict.items():
        if include_re is not None:
            if include_re.match(k) is None:
                continue
        if exclude_re is not None:
            if exclude_re.match(k) is not None:
                continue 
        res_dict[k] = p
    return res_dict


def train(config_path,
          model_dir,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          pretrained_path=None,
          pretrained_include=None,
          pretrained_exclude=None,
          freeze_include=None,
          freeze_exclude=None,
          multi_gpu=False,
          measure_time=False,
          resume=False):
    """train a VoxelNet model specified by a config file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dir = str(Path(model_dir).resolve())
    if create_folder:
        if Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    model_dir = Path(model_dir)
    if not resume and model_dir.exists():
        raise ValueError("model dir exists and you don't specify resume.")
    model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)
    with (model_dir / config_file_bkp).open("w") as f:
        f.write(proto_str)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, measure_time).to(device)
    # if train_cfg.enable_mixed_precision:
    #     net.half()
    #     net.metrics_to_float()
    #     net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    print("num parameters:", len(list(net.parameters())))
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    if pretrained_path is not None:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = filter_param_dict(pretrained_dict, pretrained_include, pretrained_exclude)
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                new_pretrained_dict[k] = v        
        print("Load pretrained parameters:")
        for k, v in new_pretrained_dict.items():
            print(k, v.shape)
        model_dict.update(new_pretrained_dict) 
        net.load_state_dict(model_dict)
        freeze_params_v2(dict(net.named_parameters()), freeze_include, freeze_exclude)
        net.clear_global_step()
        net.clear_metrics()
    if multi_gpu:
        net_parallel = torch.nn.DataParallel(net)
    else:
        net_parallel = net
    optimizer_cfg = train_cfg.optimizer
    loss_scale = train_cfg.loss_scale_factor
    fastai_optimizer = optimizer_builder.build(
        optimizer_cfg,
        net,
        mixed=False,
        loss_scale=loss_scale)
    if loss_scale < 0:
        loss_scale = "dynamic"
    if train_cfg.enable_mixed_precision:
        max_num_voxels = input_cfg.preprocess.max_number_of_voxels * input_cfg.batch_size
        assert max_num_voxels < 65535, "spconv fp16 training only support this"
        from apex import amp
        net, amp_optimizer = amp.initialize(net, fastai_optimizer,
                                        opt_level="O2",
                                        keep_batchnorm_fp32=True,
                                        loss_scale=loss_scale
                                        )
        net.metrics_to_float()
    else:
        amp_optimizer = fastai_optimizer
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [fastai_optimizer])
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, amp_optimizer,
                                              train_cfg.steps)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    if multi_gpu:
        num_gpu = torch.cuda.device_count()
        print(f"MULTI-GPU: use {num_gpu} gpu")
        collate_fn = merge_second_batch_multigpu
    else:
        collate_fn = merge_tracking_second_batch
        num_gpu = 1

    ######################
    # PREPARE INPUT
    ######################
    train_batch = input_cfg.preprocess.num_workers
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        tracking=True,
        batch=train_batch,
        multi_workers=True,
        multi_gpu=multi_gpu)
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        tracking=True,
        multi_workers=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size * num_gpu,
        shuffle=False,
        num_workers=input_cfg.preprocess.num_workers * num_gpu,
        pin_memory=False,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=not multi_gpu)
    '''
    Previous frame t-5 -> t-1
    Current frame t-4 -> t
    Batch size is 4 for each frame
    '''
    eval_batch = 1 + 1
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size, # only support multi-gpu train
        shuffle=False,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_tracking_second_batch)
    ######################
    # TRAINING
    ######################
    model_logging = SimpleModelLog(model_dir)
    model_logging.open()
    model_logging.log_text(proto_str + "\n", 0, tag="config")
    start_step = net.get_global_step()
    total_step = train_cfg.steps
    t = time.time()
    steps_per_eval = train_cfg.steps_per_eval
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch
    
    amp_optimizer.zero_grad()
    step_times = []
    step = start_step
    try:
        while True:
            if clear_metrics_every_epoch:
                net.clear_metrics()   
            train_example_list = []
            for train_sample in dataloader:
                '''
                Handle multi-batch here
                '''
                # Collect batches for tracking
                if len(train_example_list) % train_batch != 0 or len(train_example_list) == 0:
                    train_example_list.append(train_sample)
                    continue
                else:
                    example = train_example_list[ :-1] 
                    example_2 = train_example_list[1: ]
                    # Handle scene change
                    scene_inds = [exp['metadata'][0]['image_idx'][:4] for exp in example]
                    scene_2_inds = [exp['metadata'][0]['image_idx'][:4] for exp in example_2]
                    if scene_inds != scene_2_inds:
                        print("Scene change.")
                        # Overwrite directly 
                        example = example_2
                    for i in range(len(example)):
                        example[i], example_2[i] = handle_frames(example[i], example_2[i], corr_size=model_cfg.rpn.corr_patch_size)
                    # train_example_list.clear()
                    train_example_list = [train_example_list[-1]]
                    train_example_list.append(train_sample)
                example = merge_list_inputs(example)
                example_2 = merge_list_inputs(example_2)

                # Handle coordinates
                # '''
                # check sampler works correct
                # '''
                # for i in range(train_batch - 1):
                #     image1 = example['labels'][i].reshape(2, 200, 176)
                #     image2 = example_2['labels'][i].reshape(2, 200, 176)
                #     image1 = np.where(image1 < 0, 0, image1)
                #     image2 = np.where(image2 < 0, 0, image2)

                #     image1 = image1[0] + image1[1]
                #     image2 = image2[0] + image2[1]

                #     import imageio
                #     imageio.imwrite("previous_frame_{}.png".format(i), image1)
                #     imageio.imwrite('current_frame_{}.png'.format(i), image2)
                # print("pair done")
                # embed()
                # done
                lr_scheduler.step(net.get_global_step())
                time_metrics = example["metrics"]
                example.pop("metrics")
                example_torch = example_convert_to_torch(example, float_dtype)
                example_2_torch = example_convert_to_torch(example_2, float_dtype)
                batch_size = example["anchors"].shape[0]

                ret_dict = net_parallel([example_torch, example_2_torch])

                # ret_dict = net_parallel(example_torch)
                cls_preds = ret_dict["cls_preds"]
                loss = ret_dict["loss"].mean()
                cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                cls_pos_loss = ret_dict["cls_pos_loss"].mean()
                cls_neg_loss = ret_dict["cls_neg_loss"].mean()
                loc_loss = ret_dict["loc_loss"]
                cls_loss = ret_dict["cls_loss"]
                
                cared = ret_dict["cared"]
                labels = example_2_torch["labels"]
                if train_cfg.enable_mixed_precision:
                    with amp.scale_loss(loss, amp_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                amp_optimizer.step()
                amp_optimizer.zero_grad()
                net.update_global_step()
                net_metrics = net.update_metrics(cls_loss_reduced,
                                                 loc_loss_reduced, cls_preds,
                                                 labels, cared)

                step_time = (time.time() - t)
                step_times.append(step_time)
                t = time.time()
                metrics = {}
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
                num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
                if 'anchors_mask' not in example_2_torch:
                    num_anchors = example_2_torch['anchors'].shape[1]
                else:
                    num_anchors = int(example_2_torch['anchors_mask'][0].sum())
                global_step = net.get_global_step()

                if global_step % display_step == 0:
                    if measure_time:
                        for name, val in net.get_avg_time_dict().items():
                            print(f"avg {name} time = {val * 1000:.3f} ms")

                    loc_loss_elem = [
                        float(loc_loss[:, :, i].sum().detach().cpu().numpy() /
                              batch_size) for i in range(loc_loss.shape[-1])
                    ]
                    metrics["runtime"] = {
                        "step": global_step,
                        "steptime": np.mean(step_times),
                    }
                    metrics["runtime"].update(time_metrics[0])
                    step_times = []
                    metrics.update(net_metrics)
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(
                        cls_pos_loss.detach().cpu().numpy())
                    metrics["loss"]["cls_neg_rt"] = float(
                        cls_neg_loss.detach().cpu().numpy())
                    if model_cfg.use_direction_classifier:
                        dir_loss_reduced = ret_dict["dir_loss_reduced"].mean()
                        metrics["loss"]["dir_rt"] = float(
                            dir_loss_reduced.detach().cpu().numpy())

                    metrics["misc"] = {
                        "num_vox": int(example_2_torch["voxels"].shape[0]),
                        "num_pos": int(num_pos),
                        "num_neg": int(num_neg),
                        "num_anchors": int(num_anchors),
                        "lr": float(amp_optimizer.lr),
                        "mem_usage": psutil.virtual_memory().percent,
                    }
                    model_logging.log_metrics(metrics, global_step)

                if global_step % steps_per_eval == 0:
                    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                                net.get_global_step())
                    net.eval()
                    result_path_step = result_path / f"step_{net.get_global_step()}"
                    result_path_step.mkdir(parents=True, exist_ok=True)
                    model_logging.log_text("#################################",
                                        global_step)
                    model_logging.log_text("# EVAL", global_step)
                    model_logging.log_text("#################################",
                                        global_step)
                    model_logging.log_text("Generate output labels...", global_step)
                    t = time.time()
                    detections = []
                    prog_bar = ProgressBar()
                    net.clear_timer()
                    prog_bar.start((len(eval_dataset) + eval_input_cfg.batch_size - 1)
                                // eval_input_cfg.batch_size)

                    '''
                    Evaluation 
                    keys: ['voxels', 'num_points', 'coordinates', 'num_voxels', 'metrics', 'calib', 'anchors', 'metadata']
                    '''
                    eval_example_list = []
                    for eval_sample in iter(eval_dataloader):
                        # Collect batches for tracking
                        if len(eval_example_list) % eval_batch != 0 or len(eval_example_list) == 0:
                            eval_example_list.append(eval_sample)
                            continue
                        else:
                            example = eval_example_list[ :-1] 
                            example_2 = eval_example_list[1: ]
                            # for i in range(len(example)):
                            #     example[i], example_2[i] = handle_frames(example[i], example_2[i])
                            eval_example_list.pop(0)
                            eval_example_list.append(eval_sample)
                        example = merge_list_inputs(example)
                        example_2 = merge_list_inputs(example_2)

                        if example['metadata'][0]['image_idx'] == '0000/000000':
                            # First frame
                            example = example_convert_to_torch(example, float_dtype)
                            example_2 = example_convert_to_torch(example_2, float_dtype)

                            detections += net([example, example]) # First frame 0000/000000 and 0000/000000
                            detections += net([example, example_2]) # Second frame 0000/000000 and 0000/000001

                        elif example['metadata'][0]['image_idx'][:4] != example_2['metadata'][0]['image_idx'][:4]:
                            # New scence
                            example_2 = example_convert_to_torch(example_2, float_dtype)
                            detections += net([example_2, example_2]) 
                        else:
                            # As usual
                            example = example_convert_to_torch(example, float_dtype)
                            example_2 = example_convert_to_torch(example_2, float_dtype)
                            detections += net([example, example_2]) 
                        
                        prog_bar.print_bar()

                    # Handle last one!
                    last_example = merge_list_inputs([eval_sample])
                    last_example = example_convert_to_torch(last_example, float_dtype)
                    detections += net([example_2, last_example])
                    
                    sec_per_ex = len(eval_dataset) / (time.time() - t)
                    model_logging.log_text(
                        f'generate label finished({sec_per_ex:.2f}/s). start eval:',
                        global_step)
                    result_dict = eval_dataset.dataset.evaluation(
                        detections, str(result_path_step))
                    for k, v in result_dict["results"].items():
                        model_logging.log_text("Evaluation {}".format(k), global_step)
                        model_logging.log_text(v, global_step)
                    model_logging.log_metrics(result_dict["detail"], global_step)
                    with open(result_path_step / "result.pkl", 'wb') as f:
                        pickle.dump(detections, f)
                    net.train()
                step += 1
                if step >= total_step:
                    break
                
            if step >= total_step:
                break
    except Exception as e:
        print(json.dumps(example["metadata"], indent=2))
        model_logging.log_text(str(e), step)
        model_logging.log_text(json.dumps(example["metadata"], indent=2), step)
        torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                    step)
        raise e
    finally:
        model_logging.close()
    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                net.get_global_step())


def evaluate(config_path,
             model_dir=None,
             result_path=None,
             ckpt_path=None,
             measure_time=False,
             batch_size=None,
             **kwargs):
    """Don't support pickle_result anymore. if you want to generate kitti label file,
    please use kitti_anno_to_label_file and convert_detection_to_kitti_annos
    in second.data.kitti_dataset.
    """
    assert len(kwargs) == 0
    model_dir = str(Path(model_dir).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_name = 'eval_results'
    if result_path is None:
        model_dir = Path(model_dir)
        result_path = model_dir / result_name
    else:
        result_path = Path(result_path)
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to eval with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, measure_time=measure_time).to(device)
    if train_cfg.enable_mixed_precision:
        net.half()
        print("half inference!")
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator

    if ckpt_path is None:
        assert model_dir is not None
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)
    batch_size = batch_size or input_cfg.batch_size
    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        tracking=True,
        batch=2,
        multi_workers=True)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_tracking_second_batch)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    detections = []
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    prep_example_times = []
    prep_times = []
    t2 = time.time()

    eval_batch = 1 + 1
    eval_example_list = []
    
    for eval_sample in iter(eval_dataloader):
        # Collect batches for tracking
        if len(eval_example_list) % eval_batch != 0 or len(eval_example_list) == 0:
            eval_example_list.append(eval_sample)
            continue
        else:
            example = eval_example_list[ :-1] 
            example_2 = eval_example_list[1: ]
            # for i in range(len(example)):
            #     example[i], example_2[i] = handle_frames(example[i], example_2[i])
            eval_example_list.pop(0)
            eval_example_list.append(eval_sample)
        example = merge_list_inputs(example)
        example_2 = merge_list_inputs(example_2)

        if measure_time:
            prep_times.append(time.time() - t2)
            torch.cuda.synchronize()
            t1 = time.time()

        if example['metadata'][0]['image_idx'] == '0000/000000':
            # First frame
            example = example_convert_to_torch(example, float_dtype)
            example_2 = example_convert_to_torch(example_2, float_dtype)
            with torch.no_grad():
                detections += net([example, example]) # First frame 0000/000000 and 0000/000000
                detections += net([example, example_2]) # Second frame 0000/000000 and 0000/000001

        elif example['metadata'][0]['image_idx'][:4] != example_2['metadata'][0]['image_idx'][:4]:
            # New scence
            example_2 = example_convert_to_torch(example_2, float_dtype)
            with torch.no_grad():
                detections += net([example_2, example_2]) 
        else:
            # As usual
            example = example_convert_to_torch(example, float_dtype)
            example_2 = example_convert_to_torch(example_2, float_dtype)
            with torch.no_grad():
                detections += net([example, example_2]) 
        bar.print_bar()
        if measure_time:
            t2 = time.time()
    # Handle last one!
    last_example = merge_list_inputs([eval_sample])
    last_example = example_convert_to_torch(last_example, float_dtype)
    detections += net([example_2, last_example])

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')
    if measure_time:
        print(
            f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms"
        )
        print(f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
    for name, val in net.get_avg_time_dict().items():
        print(f"avg {name} time = {val * 1000:.3f} ms")
    with open(result_path_step / "result.pkl", 'wb') as f:
        pickle.dump(detections, f)
    result_dict = eval_dataset.dataset.evaluation(detections,
                                                  str(result_path_step))
    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print("Evaluation {}".format(k))
            print(v)

def helper_tune_target_assigner(config_path, target_rate=None, update_freq=200, update_delta=0.01, num_tune_epoch=5):
    """get information of target assign to tune thresholds in anchor generator.
    """    
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, False)
    # if train_cfg.enable_mixed_precision:
    #     net.half()
    #     net.metrics_to_float()
    #     net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        tracking=True,
        multi_workers=True,
        multi_gpu=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=merge_tracking_second_batch,
        worker_init_fn=_worker_init_fn,
        drop_last=False)
    
    class_count = {}
    anchor_count = {}
    class_count_tune = {}
    anchor_count_tune = {}
    for c in target_assigner.classes:
        class_count[c] = 0
        anchor_count[c] = 0
        class_count_tune[c] = 0
        anchor_count_tune[c] = 0


    step = 0
    classes = target_assigner.classes
    if target_rate is None:
        num_tune_epoch = 0
    for epoch in range(num_tune_epoch):
        for example in dataloader:
            gt_names = example["gt_names"]
            for name in gt_names:
                class_count_tune[name] += 1
            
            labels = example['labels']
            for i in range(1, len(classes) + 1):
                anchor_count_tune[classes[i - 1]] += int(np.sum(labels == i))
            if target_rate is not None:
                for name, rate in target_rate.items():
                    if class_count_tune[name] > update_freq:
                        # calc rate
                        current_rate = anchor_count_tune[name] / class_count_tune[name]
                        if current_rate > rate:
                            target_assigner._anchor_generators[classes.index(name)].match_threshold += update_delta
                            target_assigner._anchor_generators[classes.index(name)].unmatch_threshold += update_delta
                        else:
                            target_assigner._anchor_generators[classes.index(name)].match_threshold -= update_delta
                            target_assigner._anchor_generators[classes.index(name)].unmatch_threshold -= update_delta
                        anchor_count_tune[name] = 0
                        class_count_tune[name] = 0
            step += 1
    for c in target_assigner.classes:
        class_count[c] = 0
        anchor_count[c] = 0
    total_voxel_gene_time = 0
    count = 0

    for example in dataloader:
        gt_names = example["gt_names"]
        total_voxel_gene_time += example["metrics"][0]["voxel_gene_time"]
        count += 1

        for name in gt_names:
            class_count[name] += 1
        
        labels = example['labels']
        for i in range(1, len(classes) + 1):
            anchor_count[classes[i - 1]] += int(np.sum(labels == i))
    print("avg voxel gene time", total_voxel_gene_time / count)

    print(json.dumps(class_count, indent=2))
    print(json.dumps(anchor_count, indent=2))
    if target_rate is not None:
        for ag in target_assigner._anchor_generators:
            if ag.class_name in target_rate:
                print(ag.class_name, ag.match_threshold, ag.unmatch_threshold)

def mcnms_parameters_search(config_path,
          model_dir,
          preds_path):
    pass


if __name__ == '__main__':
    fire.Fire()
