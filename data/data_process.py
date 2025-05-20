### code adapted from https://github.com/abhyantrika/mediainr ###

import os
import glob
import torch 
import torch.nn as nn
import torchvision
import decord

from .dataset import VideoDataset, DiffNeRVDataset, DivNeRVDataset

datasets = {
    'diff_nerv': DiffNeRVDataset,
    'div_nerv': DivNeRVDataset
}

class DataProcess():
    """
        Data processing pipeline for Video/Image data.
    """

    def __init__(self,data_path,**kwargs):
        self.data_path = data_path
        self.params = kwargs
        self.build()

    def build(self):
        if os.path.isfile(self.data_path) and self.data_path.endswith(('.mp4','.avi','.mov','.mkv')):
            self.data_set = self.load_video()
        elif os.path.isdir(self.data_path):
            self.data_set = self.load_data_dir()
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")

        self.data_shape = self.data_set.data_shape
        self.input_data_shape = self.data_set.input_data_shape
        normalize_range = self.params.get('normalize_range',[0,1])  

    def load_data_dir(self):
        pe = self.params.get("positional_encoding")

        if pe == "div_nerv":
            files = self.data_path
        else:
            files = glob.glob(self.data_path+'/*')
            files = sorted(files)
        
            if len(files)==0:
                raise ValueError(f"Empty directory: {self.data_path}")
            else:
                if not files[0].endswith(('.png','.jpg','.jpeg')):
                    raise ValueError(f"Unsupported file format: {files[0]}")
        
            self.num_frames = len(files)
        
        Dataset = datasets[pe] if pe in datasets else VideoDataset
        data_set = Dataset(files,**self.params)
        return data_set

    def load_video(self):
        pe = self.params.get("positional_encoding")
        if pe == "div_nerv":
            raise NotImplementedError("DivNeRV not supported")

        self.video_reader = decord.VideoReader(self.data_path)
        decord.bridge.set_bridge('torch')        

        self.patch_size = self.params.get('patch_size',None)
        self.group_size = self.params.get('group_size',1)
        
        Dataset = datasets[pe] if pe in datasets else VideoDataset
        data_set = Dataset(self.video_reader, **self.params)
        return data_set