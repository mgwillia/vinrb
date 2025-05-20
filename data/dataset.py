### code adapted from https://github.com/abhyantrika/mediainr ###

import torch
import torchvision
import torchvision.transforms as transforms
import math
import os
import json
from torch.utils.data import Dataset, Sampler
import collections
from PIL import Image

class VideoDataset(Dataset):
    """
        data_list: list of data items (e.g., image paths, video paths, etc.)
                   OR decord.VideoReader object.
    """

    def __init__(self, data_list,**kwargs):
        self.data_list = data_list
        self.group_size = kwargs.get('group_size',1)
        self.patch_size = kwargs.get('patch_size',None)
        self.crop_size = kwargs.get('crop_size',None)
        self.max_frames = kwargs.get('max_frames',None)
        self.normalize_range = kwargs.get('normalize_range',[0,1])
        self.net_type = kwargs.get('net', 'mlp')
        self.cached = kwargs.get('cached', 'none')
        nerv_patch_size = kwargs.get('nerv_patch_size', [1, -1, -1])

        if self.max_frames is not None:
            self.data_list = self.data_list[:self.max_frames]
        self.num_groups = len(self.data_list)//self.group_size

        if isinstance(self.data_list, list):
            self.dir_data = True
            self.input_data_shape = torchvision.io.read_image(self.data_list[0]).shape
        else:
            self.dir_data = False
            self.input_data_shape = self.data_list[0].permute(0,3,1,2).shape

        if nerv_patch_size is not None and self.cached == 'patch':
            self.data_shape = [3, nerv_patch_size[-2], nerv_patch_size[-1]]
        else:
            self.data_shape = self.input_data_shape

        self.input_data_shape = list(self.input_data_shape)
        if self.crop_size is not None:
            self.input_data_shape[1] = self.crop_size

        self.video_size = (self.num_groups, self.input_data_shape[1], self.input_data_shape[2])
        self.nerv_patch_size = tuple(nerv_patch_size[d] if nerv_patch_size[d] != -1 else self.video_size[d] for d in range(3))

        assert all(self.video_size[d] % self.nerv_patch_size[d] == 0 for d in range(3))
        self.num_patches = tuple(self.video_size[d] // self.nerv_patch_size[d] for d in range(3))
        self.load_cache()

    def load_cache(self):
        """
        Caching the images/patches.
        """
        if self.cached == 'image' or self.cached == 'patch':
            self.image_cached = self.load_all_images()
        else:
            self.image_cached = None

        if self.cached == 'patch':
            self.patch_cached = self.load_all_patches()
            self.image_cached = None
        else:
            self.patch_cached = None

        if self.cached == 'patch':
            self.image_cached = None

    def load_image(self, idx):
        """
        For loading single image (not cached).
        """
        assert isinstance(idx, int)
        image_path = self.data_list[idx]
        img = torchvision.io.read_image(image_path)
        return img

    def load_patch(self, idx):
        """
        For loading single 3D patch (not cached).
        """
        assert isinstance(idx, tuple) and len(idx) == 3
        patches = []
        h = idx[1] * self.nerv_patch_size[1]
        w = idx[2] * self.nerv_patch_size[2]
        for dt in range(self.nerv_patch_size[0]):
            t = idx[0] * self.nerv_patch_size[0] + dt
            image = self.image_cached[t] if self.image_cached is not None else self.load_image(t)
            patch = image[:, None, h: h + self.nerv_patch_size[1], w: w + self.nerv_patch_size[2]]
            patches.append(patch)
        return torch.concatenate(patches, dim=1)

    def load_all_images(self):
        images = {}
        for t in range(self.num_groups):
            images[t] = self.load_image(t)
        return images

    def load_all_patches(self):
        patches = {}
        for t in range(self.num_patches[0]):
            for h in range(self.num_patches[1]):
                for w in range(self.num_patches[2]):
                    patches[(t, h, w)] = self.load_patch((t, h, w))
        return patches
    
    def get_image(self, idx):
        """
        For getting single image (either cached or not).
        """
        assert isinstance(idx, int)
        return self.image_cached[idx] if self.image_cached is not None else self.load_image(idx)

    def get_patch(self, idx):
        """
        For getting single 3D patch (either cached or not).
        """
        assert isinstance(idx, tuple) and len(idx) == 3
        return self.patch_cached[idx] if self.patch_cached is not None else self.load_patch(idx)

    def __len__(self):
        if self.net_type == 'nvp':
            return 1
        elif self.cached == 'patch':
            return math.prod(self.num_patches)
        else:
            return self.num_groups

    def __getitem__(self, idx):
        #breakpoint()
        data_batch = {}

        if not self.dir_data:
            group_images = self.data_list[idx*self.group_size:(idx+1)*self.group_size]
            group_images = group_images.permute(0,3,1,2)            
        else:
            #group_list = self.data_list[idx*self.group_size:(idx+1)*self.group_size]
            group_list = list(range(idx*self.group_size,(idx+1)*self.group_size))
            group_images = []
            for k_idx in group_list:
                #group_images.append(torchvision.io.read_image(k))
                if self.cached == 'patch':
                    patch_idx = (
                        idx // (self.num_patches[1] * self.num_patches[2]),
                        (idx % (self.num_patches[1] * self.num_patches[2])) // self.num_patches[2],
                        (idx % (self.num_patches[1] * self.num_patches[2])) % self.num_patches[2]
                    )
                    cur_patch = torch.clone(self.get_patch(patch_idx)).squeeze(1)
                    group_images.append(cur_patch)
                    data_batch['thw_idx'] = torch.tensor(patch_idx, dtype=int)
                else:
                    group_images.append(torch.clone(self.get_image(k_idx)))
            group_images = torch.stack(group_images)
        
        group_images = group_images.float()/255.0

        if self.patch_size is not None:
            raise NotImplementedError()
            #group_images = group_images.view(group_images.shape[0],group_images.shape[1],-1).mT

        if self.crop_size is not None:
            assert group_images.shape[2] < group_images.shape[3]
            assert self.crop_size < group_images.shape[2]
            crop_amount = group_images.shape[2] - self.crop_size
            group_images = group_images[:,:,int(crop_amount/2):int(crop_amount/2)+self.crop_size,:]

        
        frame_ids = list(range(idx*self.group_size,(idx+1)*self.group_size))
        
        if self.net_type == 'nvp':
            temporal_coord_idx = torch.randint(0, self.num_groups, (self.N_samples,)) 
            spatial_coord_idx = torch.randint(0, self.input_data_shape[1]*self.input_data_shape[2], (self.N_samples,)) ### sample from flat pixels, range from 0 to h*w
            nvp_img = self.nvp_data[temporal_coord_idx, spatial_coord_idx, :] 
            data_batch['features'] = nvp_img.float()/255.0
        else:
            data_batch['features'] = group_images
        data_batch['group_id'] = idx
        data_batch['frame_ids'] = frame_ids
        data_batch['norm_idx'] = [float(x / self.num_groups) for x in frame_ids]

        return data_batch

class DiffNeRVDataset(Dataset):
    def __init__(self, data_list, **kwargs):
        self.data_list = data_list
        self.frame_idx = [float(i) / (len(data_list)-1) for i in range(len(data_list))]
        self.cached = kwargs.get('cached', 'none')
        self.num_groups = len(self.data_list)
        
        if isinstance(data_list, list):
            self.dir_data = True
            self.input_data_shape = torchvision.io.read_image(data_list[0]).shape
        else:
            self.dir_data = False
            self.input_data_shape = data_list[0].permute(0,3,1,2).shape

        self.data_shape = self.input_data_shape

    def __len__(self):
        return self.num_groups

    def __getitem__(self, idx):
        prev = max(idx-1, 0)
        next = min(len(self.data_list)-1, idx+1)

        img = torchvision.io.read_image(self.data_list[idx]).float() / 255.0
        img_prev = torchvision.io.read_image(self.data_list[prev]).float() / 255.0
        img_next = torchvision.io.read_image(self.data_list[next]).float() / 255.0
        
        frame_idx = torch.tensor(self.frame_idx[idx])

        data_batch = {
            "features": torch.stack((img_prev, img, img_next)),
            "frame_ids": (prev, idx, next),
            "group_id": idx,
            "coordinates": torch.empty(0), 
            "norm_idx": frame_idx
        }

        return data_batch

class DivNeRVDataset(Dataset):
    def __init__(self, base_dir, **kwargs):
        keyframe_quality = kwargs.get("keyframe_quality", 3)
        dataset_mean = kwargs.get("dataset_mean")
        dataset_std = kwargs.get("dataset_std")
        video_name = kwargs.get("video_name")
        self.clip_size = kwargs.get("clip_size", 8)
        self.cached = kwargs.get('cached', 'none')

        self.gt_base_dir = os.path.join(base_dir, "gt")
        self.keyframe_base_dir = os.path.join(base_dir, "keyframe", f"q{keyframe_quality}")

        self.transform_rgb = transforms.Compose([transforms.ToTensor()])
        self.transform_keyframe = transforms.Compose([transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std)])

        self.vid_length_dict = collections.OrderedDict()

        with open(os.path.join(base_dir, "annotation", "video_length.json") , 'r') as fp:
            self.vid_length_dict = json.load(fp)[video_name]

        vid_dict = collections.OrderedDict()
        self.frame_count_total = 0
        self.frame_path_list = []
        for vid_name, vid_length in self.vid_length_dict.items():
            # we divide videos into consecutive video_clips
            num_clip = round(math.ceil(vid_length / self.clip_size))
            # rounded up the vid_length, in case the vid_length is not divided by the self.clip_size
            vid_length_round = num_clip * self.clip_size

            for clip_index in range(num_clip):
                # the first frame is the start_keyframe, the first frame for the next consecutive clip is the end_keyframe
                start_keyframe_index = clip_index * self.clip_size + 1
                end_keyframe_index = min(vid_length, (clip_index + 1) * self.clip_size + 1)

                vid_clip_name = "{}-{}".format(vid_name, clip_index)
                vid_dict[vid_clip_name] = {}
                vid_dict[vid_clip_name]['vid_name'] = vid_name
                vid_dict[vid_clip_name]['vid_length'] = vid_length 
                vid_dict[vid_clip_name]['keyframe_path'] = ['frame{:06d}.png'.format(start_keyframe_index), 'frame{:06d}.png'.format(end_keyframe_index)]
                frame_index_list = list(range(clip_index * self.clip_size + 1, (clip_index + 1) * self.clip_size + 1))
                # mask out the frame_index which are longer than the actual vid_length
                vid_dict[vid_clip_name]['frame_mask'] = [(frame_index <= vid_length) for frame_index in frame_index_list]
                frame_index_list = [min(frame_index, vid_length) for frame_index in frame_index_list]
                vid_dict[vid_clip_name]['backward_distance'] = [(frame_index - start_keyframe_index) / max(1, end_keyframe_index - start_keyframe_index) for frame_index in frame_index_list]
                vid_dict[vid_clip_name]['frame_path'] = ['frame{:06d}.png'.format(frame_index) for frame_index in frame_index_list]
                # normalize input_index by the original vid_length to [0, 1]
                vid_dict[vid_clip_name]['input_index'] = [(frame_index - 1) / (vid_length - 1) for frame_index in frame_index_list]
                self.frame_path_list.append([vid_clip_name, vid_name, vid_dict[vid_clip_name]['frame_path']])
                self.frame_count_total += self.clip_size

        self.vid_dict = vid_dict
        self.vid_list = sorted(list(vid_dict.keys()))
        self.frame_path_list = sorted(self.frame_path_list)

        self.input_data_shape = (3, kwargs.get("spatial_size_h"), kwargs.get("spatial_size_w"))
        self.data_shape = self.input_data_shape
        self.num_groups = len(self.vid_list)

    def __len__(self):
        return self.num_groups

    def __getitem__(self, idx):
        vid_clip_name = self.vid_list[idx]
        vid_name = self.vid_dict[vid_clip_name]['vid_name']

        frame_list = []
        for k in range(len(self.vid_dict[vid_clip_name]['frame_path'])):
            frame_path = self.vid_dict[vid_clip_name]['frame_path'][k]
            frame = Image.open(os.path.join(self.gt_base_dir, vid_name, frame_path)).convert("RGB")
            frame_list.append(self.transform_rgb(frame))
        video = torch.stack(frame_list, dim=1)

        input_index = torch.tensor(self.vid_dict[vid_clip_name]['input_index'])

        start_keyframe = self.transform_keyframe(Image.open(os.path.join(self.keyframe_base_dir, vid_name, self.vid_dict[vid_clip_name]['keyframe_path'][0])).convert("RGB"))
        end_keyframe = self.transform_keyframe(Image.open(os.path.join(self.keyframe_base_dir, vid_name, self.vid_dict[vid_clip_name]['keyframe_path'][1])).convert("RGB"))
        keyframe = torch.stack([start_keyframe, end_keyframe], dim=1)

        backward_distance = torch.tensor(self.vid_dict[vid_clip_name]['backward_distance'])
        frame_mask = torch.tensor(self.vid_dict[vid_clip_name]['frame_mask'])
        return video, input_index, keyframe, backward_distance, frame_mask 


class DivNervEvalSampler(Sampler):
    def __init__(self, data_source, skip):
        self.data_source = data_source
        self.skip = skip
        self.length = len(data_source)
    
    def __iter__(self):
        for offset in range(self.skip):
                for index in range(0, self.length, self.skip):
                    yield index + offset
    
    def __len__(self):
        return self.length