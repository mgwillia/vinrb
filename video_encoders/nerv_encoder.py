### code adapted from https://github.com/abhyantrika/mediainr ###

import numpy as np

from .inr_encoder import INREncoder
from models import NeRV
from utils import compute_nerv_dim_pe,compute_nerv_dim_enc

class NeRVEncoder(INREncoder):
    def __init__(self,data_path,positional_encoding=None,**kwargs):

        self.data_path = data_path
        self.positional_encoding = positional_encoding
        
        self.params = kwargs
        
        self.build_model()
        self.build_data_pipeline()

    def build_model(self):

        # Default configurations for the network
        self.default_config = {
            'reduce': 2,
            'lower_width': 12,
            'ks': '3_3',
            'activation': 'gelu',
        }

        net_config = {**self.default_config, **self.params}

        self.load_positional_encoder()

        ks_1, ks_2 = [int(x) for x in net_config['ks'].split('_')]
        if self.positional_encoding in ['hnerv']:
            hnerv_hw = np.prod(net_config['encoder_strides']) // np.prod(net_config['decoder_strides'])
            fc_h, fc_w = hnerv_hw, hnerv_hw
            if net_config.get('is_hinerv', False):
                fc_h *= 4
                fc_w *= 4
        elif self.positional_encoding == 'diff_nerv':
            fc_h, fc_w = 0, 0
        else:
            fc_h, fc_w = [int(x) for x in net_config['fc_hw'].split('_')]
            
        if ('target_modelsize' in net_config and net_config['target_modelsize'] > 0.0) or self.positional_encoding in ['hnerv', 'henerv']:
            if self.positional_encoding == 'nerv':
                fc_dim = compute_nerv_dim_pe(net_config['pe_levels'], net_config['target_modelsize'],  
                                        fc_h, fc_w, net_config['reduce'],
                                        ks_1, ks_2, net_config['decoder_strides'], net_config['lower_width'])
            elif self.positional_encoding in ['hnerv', 'henerv']:
                fc_dim, enc_dim2 = compute_nerv_dim_enc(net_config['encoder_strides'], net_config['encoder_dims'], 
                                                    net_config['target_modelsize'], net_config['frame_hw'], net_config['full_data_length'], 
                                                    net_config['reduce'], ks_1, ks_2, net_config['decoder_strides'], net_config['lower_width'])
                _, enc_dim2 = [int(x) for x in net_config['encoder_dims'].split('_')]
                if 'fc_dim' in net_config:
                    fc_dim = net_config['fc_dim']
            else:
                fc_dim = net_config['fc_dim']
        else:
            fc_dim = net_config.get('fc_dim')

        if self.positional_encoding in ['hnerv']:
            ch_in = enc_dim2
        elif self.positional_encoding == 'henerv':
            ch_in = net_config['pe_levels'] * 2
            net_config['t_ch_in'] = enc_dim2 * 6 * 6
        elif self.positional_encoding == 'ffnerv':
            ch_in = fc_dim
        elif self.positional_encoding in ['hinerv', 'hienerv']:
            ch_in = net_config['base_channels']
        elif self.positional_encoding == 'diff_nerv':
            ch_in = net_config['enc_dim']
        elif self.positional_encoding == 'div_nerv':
            ch_in = fc_dim
        else:
            ch_in = net_config['pe_levels'] * 2
        stem_type = net_config.get('stem_type', self.positional_encoding)

        expansion = net_config.get('expansion', None)
        block_dim = net_config.get('block_dim', fc_dim)
        dec_strds = net_config.get('decoder_strides', [])
        pe_t_manipulate_embed_length = net_config.get('manip_levels', 0) * 2

        net_config['positional_encoding'] = self.positional_encoding

        self.net = NeRV(
            ch_in=ch_in, 
            fc_h=fc_h, 
            fc_w=fc_w, 
            fc_dim=fc_dim, 
            block_dim=block_dim,            
            dec_strds=dec_strds, 
            ks_1=ks_1, 
            ks_2=ks_2,             
            reduce=net_config['reduce'],
            lower_width=net_config['lower_width'], 
            activation=net_config['activation'],
            out_bias=('tanh' if 'out_bias' not in net_config else net_config['out_bias']),
            stem_type=stem_type, 
            pe_t_manipulate_embed_length=pe_t_manipulate_embed_length,
            expansion=expansion,
            net_config=net_config
        )
        
        self.net.positional_encoder = self.positional_encoder
        self.net.positional_encoding = self.positional_encoding

    def save_model(self,file_path):
        pass

    def load_model(self,file_path):
        pass