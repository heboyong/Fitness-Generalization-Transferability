# Extract Features from Diffusion Process
# Implementation of DIFF for paper 'Diffusion Features to Bridge Domain Gap for Semantic Segmentation'
# Based on HyperFeature
# By Yuxiang Ji

import numpy as np
import os
from PIL import Image
import PIL
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    DDIMInverseScheduler,
    StableDiffusionPipeline,
    StableDiffusionSAGPipeline,
    StableDiffusionDepth2ImgPipeline
)
from transformers import (
    CLIPModel,
    CLIPTextModel,
    CLIPTokenizer
)
from archs.stable_diffusion.resnet import set_timestep, collect_feats_resnet, collect_feats_ca

"""
Functions for running the generalized diffusion process 
(either inversion or generation) and other helpers 
related to latent diffusion models. Adapted from 
Shape-Guided Diffusion (Park et. al., 2022).
https://github.com/shape-guided-diffusion/shape-guided-diffusion/blob/main/utils.py
"""


def get_tokens_embedding(clip_tokenizer, clip, device, prompt):
    tokens = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    input_ids = tokens.input_ids.to(device)
    embedding = clip(input_ids).last_hidden_state
    return tokens, embedding


def latent_to_image(vae, latent):
    latent = latent / 0.18215
    image = vae.decode(latent.to(vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    image = Image.fromarray(image)
    return image


def image_to_latent(vae, image, generator=None, mult=64, w=512, h=512):
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32)
    # remove alpha channel
    if len(image.shape) == 2:
        image = image[:, :, None]
    else:
        image = image[:, :, (0, 1, 2)]
    # (b, c, w, h)
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image / 255.0
    image = 2. * image - 1.
    image = image.to(vae.device)
    image = image.to(vae.dtype)
    return vae.encode(image).latent_dist.sample(generator=generator) * 0.18215


def get_xt_next(xt, et, at, at_next, a_skip, eta, tmask, do_adpm_steps=False, gamma_t=None):
    """
    Uses the DDIM formulation for sampling xt_next
    Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
    """
    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    if eta == 0:
        c1 = 0
    else:
        c1 = (
            eta * ((1 - a_skip) * (1 - at_next) / (1 - at)).sqrt()
        )
        # c1 = eta * (1 - a_skip).sqrt()
    c2 = torch.max((1 - at_next) - (c1 * tmask) ** 2,
                   torch.Tensor([1e-10]).to(et.device))[0].sqrt()
    # print(f'c1={c1}, c2={c2}, at={at}, at_next={at_next}, c12={c1**2}')
    if do_adpm_steps and eta > 0.0:
        cov_x_0_pred = (1 - at) / at * (1. - gamma_t)
        cov_x_0_pred_clamp = torch.clamp(cov_x_0_pred, 0., 1.)
        coeff_cov_x_0 = (at_next ** 0.5 - ((c2 ** 2)
                         * at / (1 - at)) ** 0.5) ** 2
        offset = coeff_cov_x_0 * cov_x_0_pred_clamp
        c1 = (c1 ** 2 + offset) ** 0.5
        # print(f'cov_x_0_pred={cov_x_0_pred}, coef={coeff_cov_x_0}, c1={c1}, c2={c2}, at={at}, at_next={at_next}, c12={c1**2}')
    xt_next = at_next.sqrt() * x0_t + c1 * tmask * torch.randn_like(et) + c2 * et
    # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn(1).to(device=et.device, dtype=et.dtype) * tmask * torch.randn_like(et) + c2 * et
    return x0_t, xt_next


def generalized_steps(x, model, scheduler, **kwargs):
    """
    Performs either the generation or inversion diffusion process.
    """
    seq = scheduler.timesteps
    seq = torch.flip(seq, dims=(0,))
    b = scheduler.betas
    b = b.to(x.device)

    batch_size = x.shape[0]
    x_h = x.shape[2]
    x_w = x.shape[3]
    x_c = x.shape[1]

    images = kwargs.get("images", None)

    with torch.no_grad():
        seq_next = [0] + list(seq[:-1])

        if kwargs.get("run_inversion", False):
            seq_iter = seq_next
            seq_next_iter = seq
        else:
            seq_iter = reversed(seq)
            seq_next_iter = reversed(seq_next)

        do_one_step = kwargs.get("do_one_step", False)
        if do_one_step and kwargs.get("run_inversion", False):
            seq_iter = list(seq[:])
            seq_next_iter = seq_iter
            t = torch.tensor(seq[0], dtype=torch.long, device=x.device)
            noise = torch.randn_like(x).to(x.device)
            x = scheduler.add_noise(x, noise, t)

        s_tmin = kwargs.get("s_tmin", 10)
        s_tmax = kwargs.get("s_tmax", 250)

        tmask = torch.ones((x.shape[2], x.shape[3])).to(
            device=x.device, dtype=x.dtype)

        do_with_depth = kwargs.get("do_with_depth", False)
        # if do_with_depth:
        #     images = F.interpolate(images, size=(384, 384), mode='bilinear')
        #     depth_map = depth_estimator(images).predicted_depth
        #     depth_map = depth_map[:, None, ...]
        #     depth_map = F.interpolate(
        #         depth_map, size=(x_h, x_w), mode='bilinear')
        #     depth_min, depth_max = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True), torch.amax(depth_map, dim=[1, 2, 3],
        #                                                                                           keepdim=True)
        #     depth_map = 2. * (depth_map - depth_min) / \
        #         (depth_max - depth_min) - 1.

        do_mask_steps = kwargs.get("do_mask_steps", False)
        if do_mask_steps:
            ref_masks = kwargs.get("ref_masks", None)
            ref_labels = kwargs.get("ref_labels", None)
            clip_tokenizer = kwargs.get("clip_tokenizer")
            clip = kwargs.get("clip")

            mask_min = kwargs.get("mask_min", 0)
            mask_max = kwargs.get("mask_max", 1000)

            # Mask [len(prompts) * batch_size, 1, h, w]
            masks = torch.ones((1, batch_size, 1, x.shape[2], x.shape[3])).to(
                dtype=x.dtype, device=x.device)
            prompts = [kwargs["prompt"]] * batch_size

            prompt_cnt = 1

            if ref_masks != None and ref_labels != None:
                # for label in ref_labels:
                #     # print('label  ', label, ref_labels)
                #     label = int(label.to('cpu').item())
                #     if label not in label_map.keys():
                #         continue
                #     mask = ref_masks[label, ...]  # [batch_size, h, w]
                #     if len(mask.shape) == 3:
                #         mask = mask[:, None, :, :]
                #     mask = mask.half()
                #     mask = F.interpolate(mask, size=(x_h, x_w), mode='nearest')
                #     mask = mask[None, ...]  # [1, batch_size, 1, h, w]

                #     # final [C+1, batch_size, 1, h, w]
                #     masks = torch.cat([masks, mask], dim=0)

                #     label_text = label_map[label]
                #     prompts += [label_text] * batch_size
                #     prompt_cnt += 1

                prompts += ref_labels
                mask = ref_masks.to(dtype=x.dtype, device=x.device)
                mask = mask[:, None, :, :]
                mask = F.interpolate(mask, size=(x_h, x_w), mode='nearest')
                mask = mask[None, ...]
                masks = torch.cat([masks, mask], dim=0)
                prompt_cnt += 1

                _, label_embedding = get_tokens_embedding(
                    clip_tokenizer, clip, x.device, prompts)
                uncond_embedding = kwargs["unconditional"]
                text_embedding = torch.cat(
                    [uncond_embedding, label_embedding], dim=0)

                count = torch.zeros_like(x)
                value = torch.zeros_like(x)

        xs = x
        for i, (t, next_t) in enumerate(zip(seq_iter, seq_next_iter)):
            max_i = kwargs.get("max_i", None)
            min_i = kwargs.get("min_i", None)
            if max_i is not None and i >= max_i:
                break
            if min_i is not None and i < min_i:
                continue

            t = (torch.ones(1) * t).to(x.device)
            next_t = (torch.ones(1) * next_t).to(x.device)

            at = (1 - b).cumprod(dim=0).index_select(0, t.long())
            at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())

            # Expand to the correct dim
            at, at_next = at[:, None, None, None], at_next[:, None, None, None]

            set_timestep(model, i)

            xt = xs
            if do_with_depth:
                pass
            else:
                xt_input = xt
            # xt = torch.cat((xs, depth_map), dim=1)
            cond = kwargs["conditional"]
            guidance_scale = kwargs.get("guidance_scale", -1)

            if do_mask_steps and t >= mask_min and t <= mask_max:
                if ref_masks != None and ref_labels != None:
                    xt_input = xt_input.repeat(prompt_cnt+1, 1, 1, 1)

                    et = model(
                        xt_input, t, encoder_hidden_states=text_embedding).sample
                    et_uncond = et[:batch_size, ...].repeat(prompt_cnt, 1, 1, 1)
                    et_text = et[batch_size:, ...]
                    et = et_uncond + guidance_scale * (et_text - et_uncond)
                else:
                    uncond = kwargs["unconditional"]
                    xt_input = xt_input.repeat(2, 1, 1, 1)
                    encoder_hidden_states = torch.cat([uncond, cond], dim=0)
                    et = model(
                        xt_input, t, encoder_hidden_states=encoder_hidden_states).sample
                    et_uncond = et[:batch_size, ...]
                    et_cond = et[batch_size:, ...]
                    et = et_uncond + guidance_scale * (et_cond - et_uncond)
            elif guidance_scale == -1:
                et = model(xt, t, encoder_hidden_states=cond).sample
            else:
                # If using Classifier-Free Guidance, the saved feature maps
                # will be from the last call to the model, the conditional prediction
                uncond = kwargs["unconditional"]
                xt_input = xt_input.repeat(2, 1, 1, 1)
                encoder_hidden_states = torch.cat([uncond, cond], dim=0)
                et = model(
                    xt_input, t, encoder_hidden_states=encoder_hidden_states).sample
                et_uncond = et[:batch_size, ...]
                et_cond = et[batch_size:, ...]
                et = et_uncond + guidance_scale * (et_cond - et_uncond)

            eta = kwargs.get("eta", 1.0)
            if t > s_tmin and t < s_tmax:
                eta = eta
            else:
                eta = 0.0

            if t > next_t:
                a_skip = at / at_next
            else:
                a_skip = at_next / at

            if do_mask_steps and ref_masks != None and ref_labels != None:
                if t >= mask_min and t <= mask_max:
                    
                    xts = xt.repeat(prompt_cnt, 1, 1, 1)
                    x0_ts = (xts - et * (1 - at).sqrt()) / at.sqrt()

                    c1 = (
                        eta * ((1 - a_skip) * (1 - at_next) / (1 - at)).sqrt()
                    )
                    c2 = torch.max((1 - at_next) - (c1 * tmask) ** 2,
                                   torch.Tensor([1e-10]).to(et.device))[0].sqrt()
                    xts_next = at_next.sqrt() * x0_ts + c1 * tmask * torch.randn_like(et) + c2 * et

                # [prompt_cnt*batch_size, c, h, w] * [prompt_cnt*batch_size, 1, h, w] -> [prompt_cnt, batch_size, c, h, w] -> [batch_size, c, h, w]
                value = (xts_next * masks.view(prompt_cnt*batch_size, 1, xts.shape[2], xts.shape[3])).view(
                    prompt_cnt, batch_size, x_c, x_h, x_w).sum(dim=0)
                # [prompt_cnt, batch_size, 1, h, w] -> [batch_size, 1, h, w]
                count = masks.sum(dim=0)
                xt_next = torch.where(count > 0, value / count, value)
                

            else:
                _, xt_next = get_xt_next(
                    xt, et, at, at_next, a_skip, eta, tmask)

            xs = xt_next

        return xs


def freeze_weights(weights):
    for param in weights.parameters():
        param.requires_grad = False


def init_models(
    device="cuda",
    model_id="runwayml/stable-diffusion-v1-5",
    freeze=True,
    do_with_depth=False,
):
    if 'depth' in model_id:
        do_with_depth = True
    if do_with_depth:
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            model_id,
            revision="fp16",
            torch_dtype=torch.float16,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            revision="fp16",
            torch_dtype=torch.float16,
        )
    unet = pipe.unet
    vae = pipe.vae
    clip = pipe.text_encoder
    clip_tokenizer = pipe.tokenizer
    unet.to(device)
    vae.to(device)
    clip.to(device)
    if freeze:
        freeze_weights(unet)
        freeze_weights(vae)
        freeze_weights(clip)
    return pipe, unet, vae, clip, clip_tokenizer


def get_stride_num(idxs):
    cnt = [0 for _ in range(3)]
    for [i, j] in idxs:
        cnt[i - 1] += 1
    return cnt[0], cnt[1], cnt[2]


def collect_stride_feats_with_timesteplist(unet, idxs_resnet, idxs_ca, timestep_list, do_mask_steps=False,
                                           x=None, ref_semantic_seg=None, label_map=None, guidance_scale=0.0):
    batch_size = x.shape[0]
    latent_h = [0, 0, 0, 0]
    latent_w = [0, 0, 0, 0]

    latent_feats_resnet = collect_feats_resnet(unet, idxs=idxs_resnet)
    latent_feats_resnet_idxs = []
    for idx, feat in zip(idxs_resnet, latent_feats_resnet):
        latents_feats_idxs_t = []
        for timestep in timestep_list:
            feat_t = feat[timestep]
            feat_t = feat_t[:batch_size, ...]
            latents_feats_idxs_t.append(feat_t)
        latent_feats_resnet_idxs.append(latents_feats_idxs_t)
        latent_h[idx[0]] = feat_t.shape[2]
        latent_w[idx[0]] = feat_t.shape[3]
    feats_cat_resnet_idxs = [torch.cat(feat_t, dim=1)
                             for feat_t in latent_feats_resnet_idxs]
    # print('resnet shape', feats_cat_resnet_idxs[0].shape, feats_cat_resnet_idxs[3].shape, feats_cat_resnet_idxs[6].shape, feats_cat_resnet_idxs[9].shape)

    latent_feats_ca = collect_feats_ca(unet, idxs=idxs_ca)
    latent_feats_ca_idxs = []
    for idx, feat in zip(idxs_ca, latent_feats_ca):
        latents_feats_idxs_t = []
        for timestep in timestep_list:
            feat_t = feat[timestep]
            prompt_cnt = feat_t.shape[0] // batch_size
            feat_c = feat_t.shape[2]
            # print('??', feat_t.shape, prompt_cnt, batch_size, idx, latent_h[idx[0]], latent_w[idx[0]], feat_c)
            feat_t = feat_t.view(prompt_cnt, batch_size,
                                 latent_h[idx[0]], latent_w[idx[0]], feat_c)
            feat_t = feat_t.permute(1, 0, 4, 2, 3)
            feat_t = feat_t.sum(dim=1)
            latents_feats_idxs_t.append(feat_t)
        latent_feats_ca_idxs.append(latents_feats_idxs_t)
    feats_cat_ca_idxs = [torch.cat(feat_t, dim=1)
                         for feat_t in latent_feats_ca_idxs]

    # print('ca shape', feats_cat_ca_idxs[0].shape, feats_cat_ca_idxs[3].shape, feats_cat_ca_idxs[6].shape)

    # with CrossAttn
    if len(idxs_ca) != 0 and len(idxs_resnet) != 0:
        return torch.cat(feats_cat_resnet_idxs[0:3], dim=1), torch.cat(feats_cat_resnet_idxs[3:6]+feats_cat_ca_idxs[0:3], dim=1), torch.cat(feats_cat_resnet_idxs[6:9]+feats_cat_ca_idxs[3:6], dim=1), torch.cat(feats_cat_resnet_idxs[9:12]+feats_cat_ca_idxs[6:9], dim=1)

    elif len(idxs_ca) != 0 and len(idxs_resnet) == 0:
        return None, torch.cat(feats_cat_ca_idxs[0:3], dim=1), torch.cat(feats_cat_ca_idxs[3:6], dim=1), torch.cat(feats_cat_ca_idxs[6:9], dim=1)

    # Only support idxs (0,0) (0,1) (0,2) (1,0) (1,1) (1,2) (2,0) (2,1) (2,2) (3,0) (3,1) (3,2) now. Added by  
    elif len(idxs_resnet) != 0 and len(idxs_ca) == 0:
        return torch.cat(feats_cat_resnet_idxs[0:3], dim=1), torch.cat(feats_cat_resnet_idxs[3:6], dim=1), torch.cat(feats_cat_resnet_idxs[6:9], dim=1), torch.cat(feats_cat_resnet_idxs[9:12], dim=1)


if __name__ == '__main__':
    pass
