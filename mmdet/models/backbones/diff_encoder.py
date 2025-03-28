# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from .diff.src.models.diff import DIFFEncoder


@MODELS.register_module()
class DIFF(BaseModule):
    def __init__(self,
                 init_cfg=None,
                 diff_config=dict(  aggregation_type="direct_aggregation",
                                    fine_type = 'deep_fusion',
                                    projection_dim=[2048, 2048, 1024, 512],
                                    projection_dim_x4=256,
                                    model_id="../stable-diffusion-2-1-base",
                                    diffusion_mode="inversion",
                                    input_resolution=[512, 512],
                                    prompt="",
                                    negative_prompt="",
                                    guidance_scale=-1,
                                    scheduler_timesteps=[50, 25],
                                    save_timestep=[0],
                                    num_timesteps=1,
                                    idxs_resnet=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [
                                        1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]],
                                    idxs_ca=[[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]],
                                    s_tmin=10,
                                    s_tmax=250,
                                    do_mask_steps=True,
                                    classes=('bicycle', 'bus', 'car', 'motorcycle',
                                            'person', 'rider', 'train', 'truck')
                                  )
                ):
        super().__init__(init_cfg)

        self.diff_model = None
        assert diff_config is not None
        self.diff_config = diff_config
        self.diff_model = DIFFEncoder(config=self.diff_config)

    def forward(self, x, ref_masks=None, ref_labels=None):
        x = self.imagenet_to_stable_diffusion(x)
        x = self.diff_model(x.to(dtype=torch.float16), ref_masks, ref_labels)
        return x

    def init_weights(self):
        pass

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        pass

    def imagenet_to_stable_diffusion(self, tensor):
        """
        将 ImageNet 格式的张量转换为 Stable Diffusion 格式。

        参数:
        tensor (torch.Tensor): 形状为 (N, C, H, W)，已按照 ImageNet 格式标准化。

        返回:
        torch.Tensor: 形状为 (N, C, H, W)，标准化到 [-1, 1] 范围。
        """
        # ImageNet 的均值和标准差
        mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(tensor.device)

        # 逆标准化：将张量从 ImageNet 格式恢复到 [0, 255] 范围
        tensor = tensor * std + mean

        # 转换到 [0, 1] 范围
        tensor = tensor / 255.0

        # 转换到 [-1, 1] 范围
        tensor = tensor * 2.0 - 1.0

        return tensor
