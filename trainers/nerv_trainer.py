### code adapted from https://github.com/abhyantrika/mediainr ###

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import tqdm
import random
import time
import copy
import os
from typing import Optional
import gc
from torchvision.utils import save_image
from PIL import Image
import csv
import math
import einops

from data.dataset import DivNervEvalSampler
from utils import losses, helpers, coord_utils, compress_bitstream, decompress_bitstream, set_zero, checkpoint, adjust_lr_linear_warmup_cosine_annealing
from utils.hooks import get_flops_post_iter


class NeRVTrainer:
    def __init__(self, encoder, data_pipeline, val_loader=None, batch_size:int=1,num_iters:int=1,loss:str='mse',extra_params:dict={}, **train_params:Optional[dict]):

        """
        Nirvana Trainer for training the Nirvana Encoder.
        :param encoder: Encoder to train.
        :param data_pipeline: Data pipeline for the encoder.
        :param val_loader: Validation loader.
        :param batch_size: Batch size.
        :param train_params: Training parameters. Options include:
            - lr (float): Learning rate. Default is 5e-4.
            - loss (str): Loss function. Default is 'mse'.
            - num_iters_first (int): Number of iterations for the first epoch. Default is 10000.
            - num_iters (int): Number of iterations for the remaining epochs. Default is 2500.
        
        :save options: params for model saving. 
            - save_path (str): Path to save the model. Default is 'output/nirvana_output/'.
            - skip_save (bool): Skip saving the outputs. Default is False.
            - skip_save_model (bool): Skip saving the model parameters. Default is False.

        """
        self.encoder = encoder
        self.model = encoder.net
        self.data_pipeline = data_pipeline
        self.num_iters = num_iters
        self.loss_type = loss
        self.train_params = train_params
        self.batch_size = batch_size

        self.save_path = self.train_params.get('save_path','output/nerv_output/')
        self.skip_save = self.train_params.get('skip_save',False)
        self.skip_save_model = self.train_params.get('skip_save_model',False)
        
        num_workers = self.train_params.get('num_workers',0)

        self.is_hnerv = self.encoder.params.get('is_hnerv', False)
        if not self.is_hnerv:
            self.is_hnerv = self.train_params.get('is_hnerv', False)

        self.is_henerv = self.encoder.params.get('is_henerv', False)
        if not self.is_henerv:
            self.is_henerv = self.train_params.get('is_henerv', False)

        self.is_diff_nerv = self.encoder.params.get('is_diff_nerv', False)
        if not self.is_diff_nerv:
            self.is_diff_nerv = self.train_params.get('is_diff_nerv', False)

        self.is_div_nerv = self.encoder.params.get('is_div_nerv', False)
        if not self.is_div_nerv:
            self.is_div_nerv = self.train_params.get('is_div_nerv', False)

        self.is_hinerv = self.encoder.params.get('is_hinerv', False)
        if not self.is_hinerv:
            self.is_hinerv = self.train_params.get('is_hinerv', False)

        self.agg_ind = self.encoder.params.get('agg_ind', None)
        if  self.agg_ind == None:
            self.agg_ind  = self.train_params.get('agg_ind', None)

        self.is_ffnerv = self.encoder.params.get('is_ffnerv', False)
        if not self.is_ffnerv:
            self.is_ffnerv = self.train_params.get('is_ffnerv', False)

        self.loss_weight = self.encoder.params.get('loss_weight', None)

        if self.loss_weight == None:
            self.loss_weight  = self.train_params.get('loss_weight', None)
        
        self.do_eval = self.train_params.get('do_eval', False)
        self.quant_level = self.train_params.get('quant_level', None)

        self.lr = extra_params.get('lr', 5e-4)
        self.warmup_lr = extra_params.get('warmup_lr', self.lr*0.01)
        self.min_lr = extra_params.get('min_lr', self.warmup_lr)
        self.warmup_epochs = extra_params.get('warmup_epochs', 0)

        self.checkpoint_path = extra_params.get('checkpoint_path', '')
        if self.checkpoint_path == '':
            self.checkpoint_path = self.train_params.get('checkpoint_path', '')
            
        self.checkpoint_freq = extra_params.get('checkpoint_freq', -1)
        if self.checkpoint_freq == -1:
            self.checkpoint_freq = self.train_params.get('checkpoint_freq', -1)

        self.resume = extra_params.get('resume', False)
        if not self.resume:
            self.resume = self.train_params.get('resume', False)

        collate_fn = helpers.custom_collate_fn
        if self.is_div_nerv:
            collate_fn = helpers.div_nerv_collate_fn

        sampler = None
        self.num_frames = len(data_pipeline.data_set)
        self.dataloader = torch.utils.data.DataLoader(data_pipeline.data_set,batch_size=batch_size,
                                                      shuffle=True,num_workers=num_workers,
                                                      sampler=sampler,collate_fn=collate_fn)
        
        if val_loader is None:
            if self.is_div_nerv:
                vid_length_dict = data_pipeline.data_set.vid_length_dict
                # assume all same number of frames
                total_frames = vid_length_dict[next(iter(vid_length_dict))]
                skip = round(math.ceil(total_frames / data_pipeline.data_set.clip_size))
                sampler = DivNervEvalSampler(data_pipeline.data_set, skip=skip)

            self.eval_loader = torch.utils.data.DataLoader(data_pipeline.data_set,batch_size=batch_size,shuffle=False,num_workers=num_workers,sampler=sampler,
                                                        collate_fn=collate_fn)
        else:
            self.eval_loader = val_loader

        # Initialize optimizers, loss functions, etc.
        self.setup()

    def setup(self):
        # Initialize optimizers, loss functions, etc.
        torch.backends.cudnn.benchmark = True
        seed = self.train_params.get('seed', random.randint(1, 10000))            
        cudnn.enabled = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        cudnn.benchmark = True # set to false if deterministic
        torch.set_printoptions(precision=10)
            
        self.encoder.net = self.encoder.net.cuda()

        default_train_config = {
            'loss': 'mse',
        }

        self.train_params = {**default_train_config, **self.train_params}
        self.optimizer = torch.optim.Adam(self.encoder.net.parameters(), lr=self.lr)
        if self.warmup_epochs > 0:
            warmup_iters = self.warmup_epochs * len(self.dataloader)
            def warmup(current_step: int):
                lambda_mult = self.warmup_lr * (len(self.dataloader) + current_step * ((warmup_iters - len(self.dataloader)) / warmup_iters)) / self.lr / len(self.dataloader)
                return lambda_mult
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)
            decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_iters*len(self.dataloader), eta_min=self.min_lr)
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[self.warmup_epochs*len(self.dataloader)])
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_iters*len(self.dataloader), eta_min=self.min_lr)

        self.loss_fn = self.load_loss_fn(self.loss_type)

        if self.is_ffnerv:
            self.len_agg = len(self.agg_ind)

    def load_loss_fn(self, loss_name:str):
        if loss_name == 'mse':
            return losses.mse()
        elif loss_name == 'l1':
            return losses.l1()
        elif loss_name == 'psnr':
            return losses.psnr()
        else:
            raise ValueError(f"Invalid loss function: {loss_name}")
        
    def _predict_with_flow(self, idx:int, output:tuple, key_frames:list):
        # update frame buffer
        head_outputs, flow_outputs = output
        key_frames[idx] = head_outputs[:,0:3,:,:].detach().clone().half()
        # flow-guided frame aggregation
        # indexing for batched warping
        agg_idx = idx.repeat(1, self.len_agg) + torch.tensor(self.agg_ind)
        agg_idx = torch.where(agg_idx >= 0, agg_idx, self.num_frames)
        agg_idx = torch.where(agg_idx < self.num_frames, agg_idx, self.num_frames)
        agg_frames = key_frames[agg_idx.long()].to(torch.float32)
        agg_frames = agg_frames.reshape(-1, 3, agg_frames.shape[-2], agg_frames.shape[-1])
        flows = flow_outputs[:,:2*(self.len_agg),:,:]
        flows = flows.reshape(-1, 2, flows.shape[-2], flows.shape[-1])
        # warping
        agg_frames = helpers.resample(agg_frames, flows).unsqueeze(0)
        agg_frames = agg_frames.reshape(-1, self.len_agg, 3, agg_frames.shape[-2], agg_frames.shape[-1])
        # first aggregation
        wei1 = flow_outputs[:,(self.len_agg)*2:,:,:]
        scale_factor = (head_outputs.shape[-2] // flow_outputs.shape[-2], head_outputs.shape[-1] // flow_outputs.shape[-1])
        wei1 = torch.nn.functional.interpolate(wei1, scale_factor=scale_factor, mode='nearest')
        wei1 = F.softmax(wei1, dim=1)
        # the aggregated frame
        agg_frame = torch.sum(agg_frames * wei1.unsqueeze(2), dim=1, keepdim=True)
        # second aggregation
        agg_frames = torch.cat([head_outputs[:,0:3,:,:].unsqueeze(1), agg_frame],dim=1)  
        wei2 = F.softmax(head_outputs[:,3:5,:,:], dim=1).unsqueeze(2)
        # aggregated frame, independent frame, final frame
        output_list = [agg_frame.squeeze(1), head_outputs[:,0:3,:,:], torch.sum(agg_frames * wei2, dim=1)]
        return output_list

    def _predict_with_flow_hinerv(self, idx:int, output:tuple, key_frames:list):
        raise NotImplementedError()

    def train(self, metrics_out=None, prefix=None):
        os.environ["KINETO_LOG_LEVEL"] = "5" # supress ops counter logs
        
        training_time = 0.0
        self.model.train()
        #best_model_state = copy.deepcopy(self.model.state_dict())
        best_psnr = 0.0
        start_epoch = 0
        flops = 0
        metrics = {}

        if prefix is not None:
            model_path = f'{prefix}-latest.pth'
        else:
            model_path = f'latest.pth'

        if self.resume:
            print(f'attempt resume {os.path.join(self.checkpoint_path, model_path)}')
            if os.path.exists(os.path.join(self.checkpoint_path, model_path)):
                start_epoch, training_time = checkpoint.load_checkpoint(os.path.join(self.checkpoint_path, model_path), self.model, self.optimizer, self.scheduler)
                if start_epoch == self.num_iters:
                    self.save_artefacts(self.save_path)
                    return training_time, metrics

        iteration = tqdm.tqdm(range(start_epoch, self.num_iters))
        
        print(self.model, flush=True)

        # get initial metrics
        epoch_metrics = self.eval()

        metrics[0] = {"training_time": start_epoch, "flops": 0, "psnr": epoch_metrics['psnr'], "msssim": epoch_metrics['msssim']}
        if metrics_out:
            with open(metrics_out, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([0] + list(metrics[0].values()))
        
        if self.is_ffnerv:
            if self.is_hinerv:
                key_frames = torch.cuda.FloatTensor(self.num_frames+(16*9), 3, self.data_pipeline.data_set.data_shape[1], self.data_pipeline.data_set.data_shape[2]).fill_(0).half()
            else:
                key_frames = torch.cuda.FloatTensor(self.num_frames+1, 3, self.data_pipeline.data_set.data_shape[1], self.data_pipeline.data_set.data_shape[2]).fill_(0).half()
            for _, batch in enumerate(self.dataloader):
                if self.is_div_nerv:
                    pass
                else:
                    features = batch['features'].cuda().squeeze(1)
                    norm_idx = batch['norm_idx']
                    times = norm_idx.cuda()

                model_input = {
                    'patch_mode': False,
                    'hinerv_sizes': {
                        'idx_max': (600, 1, 1)
                    },
                    'frame_id': batch['frame_ids'].cuda()
                }

                if self.is_hnerv or self.is_diff_nerv or self.is_henerv:
                    model_input['image'] = features
                if self.is_div_nerv:
                    video, norm_idx, keyframe, backward_distance, frame_mask = batch
                    video, norm_idx, keyframe, backward_distance, frame_mask = video.cuda(), norm_idx.cuda(), \
                                                                                keyframe.cuda(), backward_distance.cuda(), frame_mask.cuda()
                    model_input['t'] = norm_idx
                    model_input['keyframe'] = keyframe
                    model_input['backward_distance'] = backward_distance
                elif self.is_hinerv:
                    patch_idx = batch['thw_idx'].cuda()
                    hinerv_sizes = {
                        'idx_max': self.dataloader.dataset.num_patches,
                        'patch_size': self.dataloader.dataset.nerv_patch_size
                    }
                    model_input['t'] = patch_idx
                    model_input['hinerv_sizes'] = hinerv_sizes
                    model_input['patch_mode'] = True
                else:
                    model_input['t'] = times
                head_outputs, _ = self.model(model_input)
            
                if self.is_hinerv:
                    times = (patch_idx[:,0]*hinerv_sizes['idx_max'][1]*hinerv_sizes['idx_max'][2] + \
                        patch_idx[:,1]*hinerv_sizes['idx_max'][2] + \
                        patch_idx[:,2])
                    key_frames[times] = head_outputs[:,0:3,:,:].detach().clone().half()
                else:
                    key_frames[torch.round(times*self.num_frames).long()] = head_outputs[:,0:3,:,:].detach().clone().half()

        for iter in iteration:
            for i, batch in enumerate(self.dataloader):
                if i == 0 and iter == 0:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        with_flops=True
                    ) as prof:
                        
                        if self.is_div_nerv:
                            pass
                        else:
                            features = batch['features'].cuda().squeeze(1)
                            norm_idx = batch['norm_idx']
                            times = norm_idx.cuda()

                        model_input = {
                            'patch_mode': False,
                            'hinerv_sizes': {
                                'idx_max': (600, 1, 1)
                            },
                            'frame_id': batch['frame_ids'].cuda()
                        }

                        if self.is_hnerv or self.is_diff_nerv or self.is_henerv:
                            model_input['image'] = features
                        if self.is_div_nerv:
                            video, norm_idx, keyframe, backward_distance, frame_mask = batch
                            video, norm_idx, keyframe, backward_distance, frame_mask = video.cuda(), norm_idx.cuda(), \
                                                                                        keyframe.cuda(), backward_distance.cuda(), frame_mask.cuda()
                            model_input['t'] = norm_idx
                            model_input['keyframe'] = keyframe
                            model_input['backward_distance'] = backward_distance
                        elif self.is_hinerv:
                            patch_idx = batch['thw_idx'].cuda()
                            hinerv_sizes = {
                                'idx_max': self.dataloader.dataset.num_patches,
                                'patch_size': self.dataloader.dataset.nerv_patch_size
                            }
                            model_input['t'] = patch_idx
                            model_input['hinerv_sizes'] = hinerv_sizes
                            model_input['patch_mode'] = True
                        else:
                            model_input['t'] = times
                        
                        start_time = time.time()
                        output = self.model(model_input)

                        if self.is_ffnerv:
                            if self.is_hinerv:
                                idx = (patch_idx[:,0]*hinerv_sizes['idx_max'][1]*hinerv_sizes['idx_max'][2] + \
                                        patch_idx[:,1]*hinerv_sizes['idx_max'][2] + \
                                        patch_idx[:,2])
                                output_list = self._predict_with_flow_hinerv(idx.cpu(), output, key_frames)
                            else:
                                idx = torch.round(norm_idx * self.num_frames).long()
                                output_list = self._predict_with_flow(idx, output, key_frames)
                            
                            target_list = [F.adaptive_avg_pool2d(features, x.shape[-2:]) for x in output_list]
                            loss_list = [self.loss_fn(output, target) for output, target in zip(output_list, target_list)]
                            # weighted loss function
                            loss_list = [loss_list[i] * (self.loss_weight if i < len(loss_list)- 1 else 1) for i in range(len(loss_list))]
                            loss = sum(loss_list)
                            preds = output_list[-1] # "final" output frame of FFNeRV
                        elif self.is_diff_nerv:
                            preds = output
                            features = F.adaptive_avg_pool2d(features[:, 1, :, :, :], preds.shape[-2:])
                            loss = self.loss_fn(preds, features) 
                        elif self.is_div_nerv:
                            B, C, D, H, W = video.size()
                            pred = output.permute(0, 2, 1, 3, 4).contiguous().view(B*D, C, H, W)
                            target = video.permute(0, 2, 1, 3, 4).contiguous().view(B*D, C, H, W)
                            frame_mask = frame_mask.view(-1)
                            pred = pred[frame_mask]
                            target = target[frame_mask].detach()

                            features = target
                            preds = pred
                            loss = self.loss_fn(preds, features)
                            lr = adjust_lr_linear_warmup_cosine_annealing(self.optimizer, iter, i, len(self.dataloader), self.lr, self.warmup_epochs, self.num_iters)
                        else:
                            preds = output
                            loss = self.loss_fn(preds,features)
                    
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()

                        training_time += (time.time() - start_time)

                        psnr = helpers.psnr(features,preds)
                        if self.is_div_nerv:
                            # Does not make sense to report PSNR for patches, only for whole image
                            iteration.set_description(f"Training {i}/{len(self.dataloader)}, not reporting PSNR because DivNerv uses patches")
                        else:
                            iteration.set_description(f"Loss: {loss.item():.4f}, PSNR: {psnr:.4f}, LR: {self.scheduler.get_last_lr()[0]}",refresh=True)

                    flops_3d = get_flops_post_iter()
                    prof_flops = sum(evt.flops for evt in prof.events())

                    flops += prof_flops + flops_3d

                    print(prof.key_averages().table(sort_by="cuda_time_total"))
                else:
                    if self.is_div_nerv:
                        pass
                    else:
                        features = batch['features'].cuda().squeeze(1)
                        norm_idx = batch['norm_idx']
                        times = norm_idx.cuda()

                    model_input = {
                        'patch_mode': False,
                        'hinerv_sizes': {
                            'idx_max': (600, 1, 1)
                        },
                        'frame_id': batch['frame_ids'].cuda()
                    }

                    if self.is_hnerv or self.is_diff_nerv or self.is_henerv:
                        model_input['image'] = features
                    if self.is_div_nerv:
                        video, norm_idx, keyframe, backward_distance, frame_mask = batch
                        video, norm_idx, keyframe, backward_distance, frame_mask = video.cuda(), norm_idx.cuda(), \
                                                                                    keyframe.cuda(), backward_distance.cuda(), frame_mask.cuda()
                        model_input['t'] = norm_idx
                        model_input['keyframe'] = keyframe
                        model_input['backward_distance'] = backward_distance
                    elif self.is_hinerv:
                        patch_idx = batch['thw_idx'].cuda()
                        hinerv_sizes = {
                            'idx_max': self.dataloader.dataset.num_patches,
                            'patch_size': self.dataloader.dataset.nerv_patch_size
                        }
                        model_input['t'] = patch_idx
                        model_input['hinerv_sizes'] = hinerv_sizes
                        model_input['patch_mode'] = True
                    else:
                        model_input['t'] = times

                    start_time = time.time()
                    output = self.model(model_input)

                    if self.is_ffnerv:
                        if self.is_hinerv:
                            idx = (patch_idx[:,0]*hinerv_sizes['idx_max'][1]*hinerv_sizes['idx_max'][2] + \
                                    patch_idx[:,1]*hinerv_sizes['idx_max'][2] + \
                                    patch_idx[:,2])
                            output_list = self._predict_with_flow_hinerv(idx.cpu(), output, key_frames)
                        else:
                            idx = torch.round(norm_idx * self.num_frames).long()
                            output_list = self._predict_with_flow(idx, output, key_frames)
                        
                        target_list = [F.adaptive_avg_pool2d(features, x.shape[-2:]) for x in output_list]
                        loss_list = [self.loss_fn(output, target) for output, target in zip(output_list, target_list)]
                        # weighted loss function
                        loss_list = [loss_list[i] * (self.loss_weight if i < len(loss_list)- 1 else 1) for i in range(len(loss_list))]
                        loss = sum(loss_list)
                        preds = output_list[-1] # "final" output frame of FFNeRV
                    elif self.is_diff_nerv:
                        preds = output
                        features = F.adaptive_avg_pool2d(features[:, 1, :, :, :], preds.shape[-2:])
                        loss = self.loss_fn(preds, features) 
                    elif self.is_div_nerv:
                        B, C, D, H, W = video.size()
                        pred = output.permute(0, 2, 1, 3, 4).contiguous().view(B*D, C, H, W)
                        target = video.permute(0, 2, 1, 3, 4).contiguous().view(B*D, C, H, W)
                        frame_mask = frame_mask.view(-1)
                        pred = pred[frame_mask]
                        target = target[frame_mask].detach()

                        features = target
                        preds = pred
                        loss = self.loss_fn(preds, features)
                        lr = adjust_lr_linear_warmup_cosine_annealing(self.optimizer, iter, i, len(self.dataloader), self.lr, self.warmup_epochs, self.num_iters)
                    else:
                        preds = output
                        loss = self.loss_fn(preds,features)
                
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    training_time += (time.time() - start_time)

                    psnr = helpers.psnr(features,preds)
                    if self.is_div_nerv:
                        # Does not make sense to report PSNR for patches, only for whole image
                        iteration.set_description(f"Training {i}/{len(self.dataloader)}, not reporting PSNR because DivNerv uses patches")
                    else:
                        iteration.set_description(f"Loss: {loss.item():.4f}, PSNR: {psnr:.4f}, LR: {self.scheduler.get_last_lr()[0]}",refresh=True)

                    """
                    if psnr > best_psnr:
                        best_psnr = psnr
                        for k, v in self.model.state_dict().items():
                            if k not in best_model_state:
                                best_model_state[k] = v
                            best_model_state[k].copy_(v)   
                        best_opt_state = copy.deepcopy(self.optimizer.state_dict())
                    """
            
            if self.do_eval or iter == self.num_iters - 1 or (self.checkpoint_freq != -1 and (iter + 1) % self.checkpoint_freq == 0):
                del output
                del preds
                del features
                del times
                if self.is_ffnerv:
                    del output_list
                    del target_list
                    del loss_list
                gc.collect()
                torch.cuda.empty_cache()
                if self.quant_level is not None:
                    qat_state_dict = copy.deepcopy(self.model.state_dict())
                    for quant_level in sorted(self.quant_level, reverse=True):
                        # Compress bitstream
                        print(f'Compress model weights into bitstream')
                        print(f'***  Quant level: {quant_level}bits')

                        num_bytes = compress_bitstream(self.model, os.path.join(self.save_path, 'bitstreams'), quant_level)
                        num_pixels = self.data_pipeline.data_set.num_groups * self.data_pipeline.data_set.input_data_shape[1] * self.data_pipeline.data_set.input_data_shape[2]
                        bits_per_pixel = num_bytes * 8 / num_pixels

                        print(f'Compressed model size: {num_bytes / 10**6:.2f}MB')
                        print(f'Bits Per Pixel (BPP): {bits_per_pixel:.4f}')

                        # Set model weights to zero for ensuring the correctness
                        set_zero(self.model)

                        # Decompress bitstream
                        print(f'Decompress model weights from bitstream')
                        decompress_bitstream(self.model, os.path.join(self.save_path, 'bitstreams'), quant_level)

                        # Evaluation
                        if self.is_ffnerv:
                            key_frames = key_frames.cpu()
                        metrics = self.eval()
                        if self.is_ffnerv:
                            key_frames = key_frames.cuda()
                        print(metrics, flush=True)

                        # Restore from the checkpoint
                        self.model.load_state_dict(qat_state_dict)
                else:
                    if self.is_ffnerv:
                        key_frames = key_frames.cpu()
                    epoch_metrics = self.eval()
                    if self.is_ffnerv:
                        key_frames = key_frames.cuda()

                metrics[iter+1] = {"training_time": training_time, "flops": flops, "psnr": epoch_metrics["psnr"], "msssim": epoch_metrics["msssim"]}

                if metrics_out:
                    with open(metrics_out, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([iter+1] + list(metrics[iter+1].values()))

            if self.checkpoint_freq != -1:
                if (iter + 1) % self.checkpoint_freq == 0:
                    filename = os.path.join(self.checkpoint_path, model_path)
                    checkpoint.save_checkpoint(epoch=iter+1, model_state_dict=self.model.state_dict(), optimizer_state_dict=self.optimizer.state_dict(), scheduler_state_dict=self.scheduler.state_dict(), training_time=training_time, save_path=filename)
            iter += 1

        self.save_artefacts(self.save_path)

        # Save the final model
        filename = os.path.join(self.checkpoint_path, model_path)        
        checkpoint.save_checkpoint(epoch=self.num_iters, model_state_dict=self.model.state_dict(), optimizer_state_dict=self.optimizer.state_dict(), scheduler_state_dict=self.scheduler.state_dict(), training_time=training_time, save_path=filename)
        #checkpoint.save_checkpoint(epoch=self.num_iters, model_state_dict=best_model_state, optimizer_state_dict=best_opt_state, training_time=training_time, save_path=filename)
    
        return training_time, metrics

    def eval_div_nerv(self, embed_quant_level=None):
        self.model.eval(embed_quant_level)
        visual_dir = f'visualize'
        os.makedirs(visual_dir, exist_ok=True)

        decoding_start = time.time()
        with torch.no_grad():
            num_patches = len(self.data_pipeline.data_set.vid_length_dict)
            clip_size = self.data_pipeline.data_set.clip_size
            h,w = self.encoder.params.get("merged_hw")
            frame_h, frame_w = self.encoder.params.get("frame_hw")
            video_name = self.encoder.params.get("video_name")
            gt_base_dir = os.path.join(self.data_pipeline.data_path, "gt")
            patch_h = self.encoder.params.get("spatial_size_h")
            patch_w = self.encoder.params.get("spatial_size_w")
            patches_x = h // patch_h
            patches_y = w // patch_w

            psnr_list = []
            msssim_list = []

            pred_patch_list = []
            gt_patch_list = []

            for i, (video, norm_idx, keyframe, backward_distance, frame_mask) in enumerate(self.eval_loader):
                video, norm_idx, keyframe, backward_distance, frame_mask = video.cuda(), norm_idx.cuda(), \
                                                                        keyframe.cuda(), backward_distance.cuda(), frame_mask.cuda()

                model_input = {
                    't': norm_idx,
                    'keyframe': keyframe,
                    'backward_distance': backward_distance,
                    'embed_quant_level': embed_quant_level,
                }
                output = self.model(model_input)
                torch.cuda.synchronize()
                B, C, T, H, W = output.shape

                pred_patch_list.append(output)

                clip_num = i // num_patches
                patch_num = i % num_patches

                # get frames clip_num * clip_size to clip_num * clip_size + (clip_size - 1) 
                for frame_idx in range(clip_num * clip_size, clip_num * clip_size + clip_size):
                    gt_patch = Image.open(os.path.join(gt_base_dir, "{}-{:02d}".format(video_name, patch_num + 1), 'frame{:06}.png'.format(frame_idx + 1))).convert("RGB")
                    gt_patch_list.append(pil_to_tensor(gt_patch).to(torch.float32))

                if i % num_patches == num_patches - 1:
                    # merge into h x w
                    pred_images = torch.stack(pred_patch_list, axis=0)
                    pred_images = pred_images.permute(1, 3, 0, 4, 5, 2) 
                    
                    pred_images = pred_images.reshape(self.batch_size, clip_size, patches_x, patches_y, patch_h, patch_w, C)
                    pred_images = pred_images.permute(0, 1, 2, 4, 3, 5, 6)
                    pred_images = pred_images.reshape(self.batch_size, clip_size, h, w, C) 
                    pred_images = pred_images.permute(0, 1, 4, 2, 3)
                    pred_images = pred_images[:, :, :, 0: frame_h, 0: frame_w]

                    pred_images = pred_images.squeeze()
                    pred_images *= 255
                    pred_images = (pred_images + 0.5).floor()
                   
                    gt_images = torch.stack(gt_patch_list, axis=0) 
                    gt_images = gt_images.permute(0, 2, 3, 1) 
                    gt_images = gt_images.reshape(num_patches, clip_size, patch_h, patch_w, C) 
                    gt_images = gt_images.permute(1,0,2,3,4) 
                    gt_images = gt_images.reshape(clip_size, patches_x, patches_y, patch_h, patch_w, C)
                    gt_images = gt_images.permute(0, 1, 3, 2, 4, 5)
                    gt_images = gt_images.reshape(clip_size, h, w, C) 
                    gt_images = gt_images.permute(0, 3, 1, 2) 
                    gt_images = gt_images[:, :, 0: frame_h, 0: frame_w] 

                    # calculate metrics
                    for frame_idx in range(clip_size):
                        pred_image = pred_images[frame_idx]
                        gt_image = gt_images[frame_idx] 

                        gt_image_cuda = gt_image.cuda()
                        pred_image_cuda = pred_image.cuda()
                        psnr_result = helpers.psnr(pred_image_cuda / 255, gt_image_cuda / 255)
                        msssim_result = msssim_result = helpers.ms_ssim((pred_image_cuda).unsqueeze(0), (gt_image_cuda).unsqueeze(0), data_range=255, size_average=True).item()

                        psnr_list.append(psnr_result)
                        msssim_list.append(msssim_result)

                    pred_patch_list = []
                    gt_patch_list = []

        psnr = np.mean(psnr_list)
        msssim = np.mean(msssim_list)
        decoding_time = time.time() - decoding_start

        return {'psnr': psnr, 'msssim': msssim, 'decoding_time': decoding_time}
            

    def eval(self, embed_quant_level=None):
        # DivNerv eval is different, need to dump all patches then merge to get true psnr/msssim
        if self.is_div_nerv:
            return self.eval_div_nerv()

        psnr_list, ssim_list = [], []
        frame_idx = 0
        key_frames = None
        decoding_time = 0
        with torch.no_grad():
            if self.is_ffnerv:
                if self.is_hinerv:
                    key_frames = torch.zeros((self.num_frames+(16*9), 3, self.data_pipeline.data_set.data_shape[1], self.data_pipeline.data_set.data_shape[2]), dtype=torch.float16)       
                else:
                    key_frames = torch.zeros((self.num_frames+1, 3, self.data_pipeline.data_set.data_shape[1], self.data_pipeline.data_set.data_shape[2]), dtype=torch.float16)       
                key_frames = key_frames.cuda()
                for _, batch in enumerate(self.eval_loader):
                    if self.is_div_nerv:
                        pass
                    else:
                        features = batch['features'].cuda().squeeze(1)
                        norm_idx = batch['norm_idx']
                        times = norm_idx.cuda()

                    model_input = {
                        'patch_mode': False,
                        'hinerv_sizes': {
                            'idx_max': (600, 1, 1)
                        },
                        'frame_id': batch['frame_ids'].cuda(),
                        'embed_quant_level': embed_quant_level,
                    }

                    if self.is_hnerv or self.is_diff_nerv or self.is_henerv:
                        model_input['image'] = features
                    if self.is_div_nerv:
                        video, norm_idx, keyframe, backward_distance, frame_mask = batch
                        video, norm_idx, keyframe, backward_distance, frame_mask = video.cuda(), norm_idx.cuda(), \
                                                                                    keyframe.cuda(), backward_distance.cuda(), frame_mask.cuda()
                        model_input['t'] = norm_idx
                        model_input['keyframe'] = keyframe
                        model_input['backward_distance'] = backward_distance
                    elif self.is_hinerv:
                        patch_idx = batch['thw_idx'].cuda()
                        hinerv_sizes = {
                            'idx_max': self.dataloader.dataset.num_patches,
                            'patch_size': self.dataloader.dataset.nerv_patch_size
                        }
                        model_input['t'] = patch_idx
                        model_input['hinerv_sizes'] = hinerv_sizes
                        model_input['patch_mode'] = True
                    else:
                        model_input['t'] = times

                    decoding_start = time.time()
                    head_outputs, _ = self.model(model_input)

                    if self.is_hinerv:
                        times = (patch_idx[:,0]*hinerv_sizes['idx_max'][1]*hinerv_sizes['idx_max'][2] + \
                            patch_idx[:,1]*hinerv_sizes['idx_max'][2] + \
                            patch_idx[:,2])
                        key_frames[times] = head_outputs[:,0:3,:,:].detach().clone().half()
                    else:
                        key_frames[torch.round(times*self.num_frames).long()] = head_outputs[:,0:3,:,:].detach().clone().half()
                    decoding_time += (time.time() - decoding_start)
            for i, batch in enumerate(self.eval_loader):
                decoding_start = time.time()
                pred_images, target_images = self.eval_forward(batch, key_frames, embed_quant_level)
                decoding_time += (time.time() - decoding_start)

                for j in range(pred_images.shape[0]):
                    psnr_list.append(helpers.psnr(target_images[j], pred_images[j]))
                    ssim_list.append(helpers.ssim(target_images[j], pred_images[j]))

                if not self.skip_save:
                    for j in range(pred_images.shape[0]):
                        name = str(frame_idx)
                        helpers.save_tensor_img(pred_images[j],filename=self.save_path+'/pred_'+name+'.png')
                        helpers.save_tensor_img(target_images[j],filename=self.save_path+'/gt_'+name+'.png')
                        frame_idx += 1
                
        print('finished eval', flush=True)
        del key_frames
        if self.is_ffnerv:
            del head_outputs
        del batch
        gc.collect()
        torch.cuda.empty_cache()

        return {'psnr': np.mean(psnr_list), 'msssim': np.mean(ssim_list), 'decoding_time': decoding_time}

    
    def eval_forward(self, batch: dict, key_frames=None, embed_quant_level=None):
        features = batch['features'].cuda().squeeze(1)
        norm_idx = batch['norm_idx']
        times = norm_idx.cuda()
        model_input = {
            'patch_mode': False,
            'hinerv_sizes': {
                'idx_max': (600, 1, 1)
            },
            'frame_id': batch['frame_ids'].cuda(),
            'embed_quant_level': embed_quant_level,
        }
        if self.is_hnerv or self.is_diff_nerv or self.is_henerv:
            model_input['image'] = features
        if self.is_hinerv:
            patch_idx = batch['thw_idx'].cuda()
            hinerv_sizes = {
                'idx_max': self.dataloader.dataset.num_patches,
                'patch_size': self.dataloader.dataset.nerv_patch_size
            }
            model_input['t'] = patch_idx
            model_input['hinerv_sizes'] = hinerv_sizes
            model_input['patch_mode'] = True
        else:
            model_input['t'] = times

        output = self.model(model_input)

        if self.is_ffnerv:
            if self.is_hinerv:
                idx = (patch_idx[:,0]*hinerv_sizes['idx_max'][1]*hinerv_sizes['idx_max'][2] + \
                        patch_idx[:,1]*hinerv_sizes['idx_max'][2] + \
                        patch_idx[:,2])
                output_list = self._predict_with_flow_hinerv(idx.cpu(), output, key_frames)
            else:
                idx = torch.round(norm_idx * self.num_frames).long()
                output_list = self._predict_with_flow(idx, output, key_frames)
            
            preds = output_list[-1] # "final" output frame of FFNeRV
        elif self.is_diff_nerv:
            preds = output
            features = F.adaptive_avg_pool2d(features[:, 1, :, :, :], preds.shape[-2:])
        else:
            preds = output
           

        if self.data_pipeline.data_set.cached =='patch':
            hp, wp = [int(vs / nps) for (vs, nps) in zip(self.data_pipeline.data_set.video_size[1:], self.data_pipeline.data_set.nerv_patch_size[1:])]
            if len(preds.shape) == 4:
                preds = preds.unsqueeze(0)
                features = features.unsqueeze(0)
            B, P, C, h, w = preds.shape
            H = hp * h
            W = wp * w
            assert P == hp * wp
            pred_images = preds.view(B, hp, wp, C, h, w).permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
            target_images = features.view(B, hp, wp, C, h, w).permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        else:
            pred_images = preds
            target_images = features

        return pred_images, target_images

    def process_outputs(self,output):
        patch_size = self.encoder.params.get('patch_size',None)

        input_data_shape = self.encoder.data_pipeline.input_data_shape
        C,H,W = input_data_shape

        if patch_size is not None:
            raise NotImplementedError()
        else:
            output = output.squeeze()
            output = output.reshape(C,H,W)

        return output

    
    def save_artefacts(self,save_dir:str,embed_quant_level=None):

        if not self.skip_save:
            with torch.no_grad():
                if self.is_ffnerv:
                    if self.is_hinerv:
                        key_frames = torch.cuda.FloatTensor(self.num_frames+(16*9), 3, self.data_pipeline.data_set.data_shape[1], self.data_pipeline.data_set.data_shape[2]).fill_(0).half()
                    else:
                        key_frames = torch.cuda.FloatTensor(self.num_frames+1, 3, self.data_pipeline.data_set.data_shape[1], self.data_pipeline.data_set.data_shape[2]).fill_(0).half()
                    for _, batch in enumerate(self.dataloader):
                        if self.is_div_nerv:
                            pass
                        else:
                            features = batch['features'].cuda().squeeze(1)
                            norm_idx = batch['norm_idx']
                            times = norm_idx.cuda()

                        model_input = {
                            'patch_mode': False,
                            'hinerv_sizes': {
                                'idx_max': (600, 1, 1)
                            },
                            'frame_id': batch['frame_ids'].cuda(),
                            'embed_quant_level': embed_quant_level,
                        }

                        if self.is_hnerv or self.is_diff_nerv or self.is_henerv:
                            model_input['image'] = features
                        if self.is_div_nerv:
                            video, norm_idx, keyframe, backward_distance, frame_mask = batch
                            video, norm_idx, keyframe, backward_distance, frame_mask = video.cuda(), norm_idx.cuda(), \
                                                                                        keyframe.cuda(), backward_distance.cuda(), frame_mask.cuda()
                            model_input['t'] = norm_idx
                            model_input['keyframe'] = keyframe
                            model_input['backward_distance'] = backward_distance
                        elif self.is_hinerv:
                            patch_idx = batch['thw_idx'].cuda()
                            hinerv_sizes = {
                                'idx_max': self.dataloader.dataset.num_patches,
                                'patch_size': self.dataloader.dataset.nerv_patch_size
                            }
                            model_input['t'] = patch_idx
                            model_input['keyframe'] = hinerv_sizes
                            model_input['patch_mode'] = True
                        else:
                            model_input['t'] = times

                        head_outputs, _ = self.model(model_input)
                        
                        if self.is_hinerv:
                            times = (patch_idx[:,0]*hinerv_sizes['idx_max'][1]*hinerv_sizes['idx_max'][2] + \
                                patch_idx[:,1]*hinerv_sizes['idx_max'][2] + \
                                patch_idx[:,2])
                            key_frames[times] = head_outputs[:,0:3,:,:].detach().clone().half()
                        else:
                            key_frames[torch.round(times*self.num_frames).long()] = head_outputs[:,0:3,:,:].detach().clone().half()

                for _, batch in enumerate(self.dataloader):
                    times = batch['norm_idx'].cuda()
                    frame_ids = batch['frame_ids']
                    features = batch['features'].cuda()
                    norm_idx = batch['norm_idx']

                    model_input = {
                        'patch_mode': False,
                        'hinerv_sizes': {
                            'idx_max': (600, 1, 1)
                        },
                        'frame_id': frame_ids.cuda(),
                        'embed_quant_level': embed_quant_level,
                    }

                    if self.is_hnerv or self.is_diff_nerv or self.is_henerv:
                        model_input['image'] = features
                    model_input['t'] = times

                    output = self.model(model_input)

                    if self.is_ffnerv:
                        if self.is_hinerv:
                            idx = (patch_idx[:,0]*hinerv_sizes['idx_max'][1]*hinerv_sizes['idx_max'][2] + \
                                    patch_idx[:,1]*hinerv_sizes['idx_max'][2] + \
                                    patch_idx[:,2])
                            output_list = self._predict_with_flow_hinerv(idx.cpu(), output, key_frames)
                        else:
                            idx = torch.round(norm_idx * self.num_frames).long()
                            output_list = self._predict_with_flow(idx, output, key_frames)
                        
                        preds = output_list[-1] # "final" output frame of FFNeRV
                    elif self.is_diff_nerv:
                        preds = output
                        features = F.adaptive_avg_pool2d(features[:, 1, :, :, :], preds.shape[-2:])
                    else:
                        preds = output

                    for i, frame_id in enumerate(frame_ids):
                        name = str(frame_id)
                        helpers.save_tensor_img(preds[i],filename=save_dir+'/pred_'+name+'.png')


    def compress(self, compression_config=None):
        if compression_config is not None:
            quant_level = compression_config.get('quant_level', None)
        else:
            quant_level = None
        if quant_level is not None:
            qat_state_dict = copy.deepcopy(self.model.state_dict())
            # Compress bitstream
            print(f'Compress model weights into bitstream', flush=True)
            print(f'***  Quant level: {quant_level}bits', flush=True)

            start = time.time()
            num_bytes = compress_bitstream(self.model, os.path.join(self.save_path, 'bitstreams'), quant_level)
            time_to_encode = time.time() - start
            num_bits = num_bytes * 8

            print(f'Compressed model size: {num_bytes / 10**6:.2f}MB', flush=True)
        else:
            time_to_encode = 0.0
            model_params = sum(p.numel() for p in self.model.parameters())
            num_bits = model_params * 32

            print(f'Compressed model size: {num_bits / 8 / 10**6:.2f}MB', flush=True)
        if quant_level is not None:
            # Restore from the checkpoint
            self.model.load_state_dict(qat_state_dict)

        return time_to_encode, num_bits


    def decode(self, compression_config=None):
        embed_quant_level=None
        if compression_config is not None:
            quant_level = compression_config.get('quant_level', None)
            embed_quant_level = quant_level
            if quant_level is not None:
                qat_state_dict = copy.deepcopy(self.model.state_dict())
                # Compress bitstream
                print(f'Compress model weights into bitstream')
                print(f'***  Quant level: {quant_level}bits')

                num_bytes = compress_bitstream(self.model, os.path.join(self.save_path, 'bitstreams'), quant_level)
                num_pixels = self.data_pipeline.data_set.num_groups * self.data_pipeline.data_set.input_data_shape[1] * self.data_pipeline.data_set.input_data_shape[2]
                bits_per_pixel = num_bytes * 8 / num_pixels

                print(f'Compressed model size: {num_bytes / 10**6:.2f}MB')
                print(f'Bits Per Pixel (BPP): {bits_per_pixel:.4f}')

                # Set model weights to zero for ensuring the correctness
                set_zero(self.model) # TODO: this must be turned off to properly benchmark hybrid, consider refactor

                # Decompress bitstream
                print(f'Decompress model weights from bitstream')
                decompress_bitstream(self.model, os.path.join(self.save_path, 'bitstreams'), quant_level)

        # Evaluation
        metrics = self.eval(embed_quant_level)

        if compression_config is not None:
            if quant_level is not None:
                # Restore from the checkpoint
                self.model.load_state_dict(qat_state_dict)

        return metrics
