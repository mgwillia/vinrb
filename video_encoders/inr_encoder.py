### code adapted from https://github.com/abhyantrika/mediainr ###

import torch 
import torch.nn as nn

from .base_video_encoder import BaseVideoEncoder
from data import DataProcess

from models.layers import positional_encoders


class INREncoder(BaseVideoEncoder):
    def __init__(self,data_path,positional_encoding=None,**kwargs):

        """
        Base INREncoder for video encoding with Implicit Neural Representations.

        Parameters:
            input_dim (tuple): Input dimension.
            output_dim (tuple): Output dimension.
            data_path (str): Path to the data.
            **kwargs: Additional configuration options.
        """
        self.data_path = data_path
        self.positional_encoding = positional_encoding

        self.params = kwargs
        
        self.build_data_pipeline()


    def build_model(self):
        pass


    def build_data_pipeline(self):
        self.data_pipeline = DataProcess(self.data_path, positional_encoding=self.positional_encoding, **self.params)


    def load_positional_encoder(self):
        manip_lbase = self.params.get('manip_lbase', None)
        manip_levels = self.params.get('manip_levels', None)

        if self.positional_encoding is None:
            self.positional_encoder = nn.Identity()
        elif self.positional_encoding == 'nerf':
            self.positional_encoder = positional_encoders.PosEncodingNeRF(dim_in=self.input_dim,sidelength=self.params['sidelength'])
        elif self.positional_encoding == 'nerv':
            self.positional_encoder = positional_encoders.PosEncodingNeRV(pe_lbase=self.params['pe_lbase'], pe_levels=self.params['pe_levels'],
                                                                        manip_lbase=manip_lbase, manip_levels=manip_levels)
        elif self.positional_encoding == 'enerv':
            fc_h, fc_w = [int(x) for x in self.params['fc_hw'].split('_')]
            self.positional_encoder = positional_encoders.PosEncodingENeRV(pe_lbase=self.params['pe_lbase'], pe_levels=self.params['pe_levels'],
                                                                           xy_lbase=self.params['xy_lbase'], xy_levels=self.params['xy_levels'],
                                                                           manip_lbase=self.params['manip_lbase'], manip_levels=self.params['manip_levels'],
                                                                           fc_h=fc_h, fc_w=fc_w)
        elif self.positional_encoding == 'henerv':
            fc_h, fc_w = [int(x) for x in self.params['fc_hw'].split('_')]
            self.positional_encoder = positional_encoders.PosEncodingHENeRV(enc_strds=self.params['encoder_strides'], enc_dims=self.params['encoder_dims'],
                                                                           xy_lbase=self.params['xy_lbase'], xy_levels=self.params['xy_levels'],
                                                                           manip_lbase=self.params['manip_lbase'], manip_levels=self.params['manip_levels'],
                                                                           fc_h=fc_h, fc_w=fc_w)
        elif self.positional_encoding == 'hienerv':
            self.positional_encoder = positional_encoders.PosEncodingHiENeRV(size=self.params['base_size'], channels=self.params['base_channels'], 
                                                                            encoding_config=self.params)
        elif self.positional_encoding == 'ffnerv':
            fc_h, fc_w = [int(x) for x in self.params['fc_hw'].split('_')]
            self.positional_encoder = positional_encoders.PosEncodingFFNeRV(t_dim=self.params['t_dim'], fc_dim=self.params['fc_dim'], fc_h=fc_h, fc_w=fc_w, wbit=self.params['wbit'],
                                                                            manip_lbase=manip_lbase, manip_levels=manip_levels)
        
        elif self.positional_encoding == 'hnerv':
            self.positional_encoder = positional_encoders.PosEncodingHNeRV(enc_strds=self.params['encoder_strides'], enc_dims=self.params['encoder_dims'], 
                                                                           modelsize=self.params['target_modelsize'], frame_hw=self.params['frame_hw'], 
                                                                           full_data_length=self.params['full_data_length'])
        elif self.positional_encoding == 'diff_nerv':
            self.positional_encoder = positional_encoders.PosEncodingDiffNeRV(enc_strds=self.params['encoder_list'], diff_enc_list=self.params['diff_enc_list'], c1_dim=self.params['enc_dim'], 
                                                                           d_dim=self.params['embed_dim'], c2_dim=self.params['dec_dim'])
        elif self.positional_encoding == 'div_nerv':
            self.positional_encoder = positional_encoders.PosEncodingDivNeRVKeyFrames(stride_list=self.params['stride_list'][::-1])
                                                                           
        elif self.positional_encoding == 'hinerv':
            self.positional_encoder = positional_encoders.PosEncodingHiNeRV(size=self.params['base_size'], channels=self.params['base_channels'], encoding_config=self.params)
        elif self.positional_encoding == 'fourier':
            pos_mapping_size = self.params.get('pos_mapping_size',256)
            pos_scale = self.params.get('pos_scale',10.0)
            self.positional_encoder = positional_encoders.PosEncodingFourier(dim_in=self.input_dim,\
                                    mapping_size=pos_mapping_size,scale=pos_scale)
        elif self.positional_encoding == 'gaussian':
            pos_mapping_size = self.params.get('pos_mapping_size',256)
            pos_scale = self.params.get('pos_scale',10.0)
            self.positional_encoder = positional_encoders.PosEncodingGaussian(dim_in=self.input_dim,\
                                    mapping_size=pos_mapping_size,scale=pos_scale)
        else:
            raise ValueError(f"Unsupported positional encoding type: {self.positional_encoding}")

    def save_model(self,file_path):
        pass

    def load_model(self,file_path):
        pass