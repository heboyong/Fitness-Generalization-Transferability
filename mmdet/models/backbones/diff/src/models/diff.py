# Extract Diffusion Features
# Implementation for DIFF for paper 'Diffusion Features to Bridge Domain Gap for Semantic Segmentation'
# By Yuxiang Ji


from PIL import Image
import random
import torch
from torch import nn
from tqdm import tqdm
import sys
import cv2
import os
import argparse
import glob
import json
import os
from omegaconf import OmegaConf
from PIL import Image
import random
import torch
from torch import nn
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from finecoder import DiftStrideUpsampleFinecoder, DiftStrideDeepFusionFinecoder


sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from archs.diffusion_extractor import DiffusionExtractor
from archs.aggregation_network import AggregationNetwork, StrideAggregationNetwork, StrideVanillaNetwork, StrideDirectAggregationNetwork
from archs.stable_diffusion.resnet import collect_dims_by_idx, collect_dims_by_stride


def load_models_stride(config, device='cuda'):
    diffusion_extractor = DiffusionExtractor(config, device)
    # print(diffusion_extractor.idxs)
    dims_by_idx = collect_dims_by_idx(diffusion_extractor.unet, idxs_resnet=diffusion_extractor.idxs_resnet, idxs_ca=diffusion_extractor.idxs_ca)
    dims_by_stride = collect_dims_by_stride(diffusion_extractor.unet, idxs_resnet=diffusion_extractor.idxs_resnet, idxs_ca=diffusion_extractor.idxs_ca)
    if config['aggregation_type'] == 'vanilla':
        aggregation_network = StrideVanillaNetwork(
            feature_dims=dims_by_idx,
            idxs=diffusion_extractor.idxs,
            device=device,
            save_timestep=config["save_timestep"],
            num_timesteps=config["num_timesteps"]
        )
    elif config['aggregation_type'] == 'aggregation':
        aggregation_network = StrideAggregationNetwork(
            projection_dim=config["projection_dim"],
            feature_dims_by_stride=dims_by_stride,
            feature_dims_by_idx=dims_by_idx,
            idxs=diffusion_extractor.idxs_resnet,
            device=device,
            save_timestep=config["save_timestep"],
            num_timesteps=config["num_timesteps"]
        )
    elif config['aggregation_type'] == 'direct_aggregation':
        aggregation_network = StrideDirectAggregationNetwork(
            projection_dim=config["projection_dim"],
            feature_dims_by_stride=dims_by_stride,
            feature_dims_by_idx=dims_by_idx,
            idxs=diffusion_extractor.idxs_resnet,
            device=device,
            save_timestep=config["save_timestep"],
            num_timesteps=config["num_timesteps"]
        )

    return config, diffusion_extractor, aggregation_network


class DIFFEncoder(nn.Module):
    def __init__(self, config, batch_size=2, mode="float", rank='cuda'):
        super().__init__()
        self.mode = mode

        self.config, self.diffusion_extractor, self.aggregation_network = load_models_stride(config)

        if config['fine_type'] == 'upsample':
            self.finecoder = DiftStrideUpsampleFinecoder()
        else:
            in_channels = self.config['projection_dim']
            hidden_dim_x4 = self.config['projection_dim_x4']
            self.finecoder = DiftStrideDeepFusionFinecoder(in_channels=in_channels, hidden_dim_x4=hidden_dim_x4)
        
        self.batch_size = batch_size

        self.change_batchsize(batch_size)

    def change_batchsize(self, batch_size):
        self.batch_size = batch_size
        self.diffusion_extractor.change_batchsize(batch_size)
    
    def change_mode(self, mode):
        self.diffusion_extractor.change_mode(mode)

    def change_precision(self, mode):
        self.mode = mode
        if mode == "float":
            self.aggregation_network.to(dtype=torch.float)
            self.finecoder.to(dtype=torch.float)
        elif mode == "half":
            self.aggregation_network.to(dtype=torch.float16)
            self.finecoder.to(dtype=torch.float16)

    def forward(self, img_tensor, ref_masks=None, ref_labels=None):
        b = img_tensor.shape[0]
        h = img_tensor.shape[2]
        w = img_tensor.shape[3]

        if b != self.batch_size:
            self.change_batchsize(b)

        with torch.no_grad():
            feats = self.diffusion_extractor.forward(img_tensor, stride_mode=True, ref_masks=ref_masks, ref_labels=ref_labels)
    
        if self.mode == "float":
            stride_hf = self.aggregation_network([feats[0].view((b, -1, h//64, w//64)).to(dtype=torch.float), 
                                                            feats[1].view((b, -1, h//32, w//32)).to(dtype=torch.float), 
                                                            feats[2].view((b, -1, h//16, w//16)).to(dtype=torch.float), 
                                                            feats[3].view((b, -1, h//8, w//8)).to(dtype=torch.float)])
        elif self.mode == "half":
            stride_hf = self.aggregation_network([feats[0].view((b, -1, h//64, w//64)), 
                                                            feats[1].view((b, -1, h//32, w//32)),
                                                            feats[2].view((b, -1, h//16, w//16)),
                                                            feats[3].view((b, -1, h//8, w//8))])
        
        # feature_fine = self.finecoder(stride_hf[0], stride_hf[1], stride_hf[2])
        feature_fine = self.finecoder(stride_hf[0], stride_hf[1], stride_hf[2], stride_hf[3])

        return feature_fine




