# Extract Cross Attention Map and Intermediate Feature from Diffusion Process
# Implementation for DIFF for paper 'Diffusion Features to Bridge Domain Gap for Semantic Segmentation'
# Based on HyperFeature
# By Yuxiang Ji


import torch
from einops import rearrange, repeat


def init_ca_resnet_func(
    unet,
    save_hidden=False,
    use_hidden=False,
    reset=True,
    save_timestep=[],
    idxs_resnet=[(1, 0)],
    idxs_ca=[(1, 0)]
):
  def new_forward_resnet(self, input_tensor, temb):
    # https://github.com/huggingface/diffusers/blob/ad9d7ce4763f8fb2a9e620bff017830c26086c36/src/diffusers/models/resnet.py#L372
    hidden_states = input_tensor

    hidden_states = self.norm1(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    if self.upsample is not None:
      input_tensor = self.upsample(input_tensor)
      hidden_states = self.upsample(hidden_states)
    elif self.downsample is not None:
      input_tensor = self.downsample(input_tensor)
      hidden_states = self.downsample(hidden_states)

    hidden_states = self.conv1(hidden_states)

    if temb is not None:
      temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
      hidden_states = hidden_states + temb

    hidden_states = self.norm2(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states)

    if self.conv_shortcut is not None:
      input_tensor = self.conv_shortcut(input_tensor)

    if save_hidden:
      if save_timestep is None or self.timestep in save_timestep:
        # if do_optim_steps:
        #   self.mt = self.beta1 * self.mt + (1 - self.beta1) * hidden_states
        #   self.vt = self.beta2 * self.vt + (1 - self.beta2) * hidden_states ** 2
        #   self.mt = self.mt / (1 - self.beta1 ** (self.steps + 1))
        #   self.vt = self.vt / (1 - self.beta2 ** (self.steps + 1))
        #   self.feats[self.timestep] = self.mt / ((self.vt ** 0.5) + 1e-8)
        #   # print(self.__class__.__name__, torch.any(torch.isnan(self.mt / ((self.vt ** 0.5) + 1e-8))))
        #   self.steps += 1
        # else: 
        self.feats[self.timestep] = hidden_states
    elif use_hidden:
      hidden_states = self.feats[self.timestep]
    output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
    return output_tensor
  
  def new_forward_ca(
    self,
    hidden_states,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    timestep=None,
    cross_attention_kwargs=None,
    class_labels=None,
  ):
    # Notice that normalization is always applied before the real computation in the following blocks.
    # 1. Self-Attention
    if self.use_ada_layer_norm:
      norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.use_ada_layer_norm_zero:
      norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
        hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
      )
    else:
      norm_hidden_states = self.norm1(hidden_states)

    cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
    attn_output = self.attn1(
      norm_hidden_states,
      encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
      attention_mask=attention_mask,
      **cross_attention_kwargs,
    )
    if self.use_ada_layer_norm_zero:
      attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = attn_output + hidden_states

    # 2. Cross-Attention
    if self.attn2 is not None:
      norm_hidden_states = (
          self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
      )
      # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
      # prepare attention mask here

      attn_output = self.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=encoder_attention_mask,
        **cross_attention_kwargs,
      )

      hidden_states = attn_output + hidden_states
      

    # 3. Feed-forward
    norm_hidden_states = self.norm3(hidden_states)

    if self.use_ada_layer_norm_zero:
      norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    ff_output = self.ff(norm_hidden_states)

    if self.use_ada_layer_norm_zero:
      ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = ff_output + hidden_states

    #### ca4
    if save_hidden:
        if save_timestep is None or self.timestep in save_timestep:
          self.feats[self.timestep] = hidden_states

    return hidden_states
  
  layers = collect_layers_resnet(unet, idxs_resnet)
  for module in layers:
    module.forward = new_forward_resnet.__get__(module, type(module))
    if reset:
      module.feats = {}
      module.timestep = None
  
  layers = collect_layers_ca(unet, idxs_ca)
  # print('    ca', len(layers))
  for module in layers:
    module.forward = new_forward_ca.__get__(module, type(module))
    if reset:
      module.feats = {}
      module.timestep = None
  


"""
Function override for Huggingface implementation of latent diffusion models
to cache features. Design pattern inspired by open source implementation 
of Cross Attention Control.
https://github.com/bloc97/CrossAttentionControl
"""
def init_resnet_func(
  unet,
  save_hidden=False,
  do_optim_steps=False,
  use_hidden=False,
  reset=True,
  save_timestep=[],
  idxs=[(1, 0)]
):
  def new_forward(self, input_tensor, temb):
    # https://github.com/huggingface/diffusers/blob/ad9d7ce4763f8fb2a9e620bff017830c26086c36/src/diffusers/models/resnet.py#L372
    hidden_states = input_tensor

    hidden_states = self.norm1(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    if self.upsample is not None:
      input_tensor = self.upsample(input_tensor)
      hidden_states = self.upsample(hidden_states)
    elif self.downsample is not None:
      input_tensor = self.downsample(input_tensor)
      hidden_states = self.downsample(hidden_states)

    hidden_states = self.conv1(hidden_states)

    if temb is not None:
      temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
      hidden_states = hidden_states + temb

    hidden_states = self.norm2(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states)

    if self.conv_shortcut is not None:
      input_tensor = self.conv_shortcut(input_tensor)

    if save_hidden:
      if save_timestep is None or self.timestep in save_timestep:
        if do_optim_steps:
          self.mt = self.beta1 * self.mt + (1 - self.beta1) * hidden_states
          self.vt = self.beta2 * self.vt + (1 - self.beta2) * hidden_states ** 2
          self.mt = self.mt / (1 - self.beta1 ** (self.steps + 1))
          self.vt = self.vt / (1 - self.beta2 ** (self.steps + 1))
          self.feats[self.timestep] = self.mt / ((self.vt ** 0.5) + 1e-8)
          # print(self.__class__.__name__, torch.any(torch.isnan(self.mt / ((self.vt ** 0.5) + 1e-8))))
          self.steps += 1
        else: 
          self.feats[self.timestep] = hidden_states
    elif use_hidden:
      hidden_states = self.feats[self.timestep]
    output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
    return output_tensor
  
  layers = collect_layers(unet, idxs)
  for module in layers:
    module.mt = 0
    module.vt = 0
    module.beta1 = 0.9
    module.beta2 = 0.99
    module.steps = 0
    module.forward = new_forward.__get__(module, type(module))
    if reset:
      module.feats = {}
      module.timestep = None

def set_timestep(unet, timestep=None):
  for name, module in unet.named_modules():
    module_name = type(module).__name__
    # print(module_name)
    module.timestep = timestep

def collect_layers_resnet(unet, idxs=None):
  layers = []
  for i, up_block in enumerate(unet.up_blocks):
    for j, module in enumerate(up_block.resnets):
      if idxs is None or (i, j) in idxs or [i, j] in idxs:
      # print(i, j, module)
        layers.append(module)
  return layers

def collect_layers_ca(unet, idxs=None):
  layers = []
  for i, up_block in enumerate(unet.up_blocks):
    if hasattr(up_block, 'attentions'):
      for j, module in enumerate(up_block.attentions):
        if idxs is None or (i, j) in idxs or [i, j] in idxs:
          layers.append(module.transformer_blocks[0])
  return layers

def collect_layers(unet, idxs=None):
  layers = []
  for i, up_block in enumerate(unet.up_blocks):
    for j, module in enumerate(up_block.resnets):
      if (i, j) in idxs or [i, j] in idxs:
      # print(i, j, module)
        layers.append(module)
  return layers

def collect_dims_by_stride(unet, idxs_resnet=None, idxs_ca=None):
  dims = [0 for _ in range(4)]
  for i, up_block in enumerate(unet.up_blocks):
      for j, module in enumerate(up_block.resnets):
          if [i, j] in idxs_resnet or (i, j) in idxs_resnet:
            # print('!!!! ', i, j, module.time_emb_proj.out_features, module)
            dims[i] += module.time_emb_proj.out_features
  for i, up_block in enumerate(unet.up_blocks):
    if hasattr(up_block, 'attentions'):
      for j, module in enumerate(up_block.attentions):
        if [i, j] in idxs_ca or (i, j) in idxs_ca:
          dims[i] += module.transformer_blocks[0].norm1.normalized_shape[0]
  return dims

def collect_dims_by_idx(unet, idxs_resnet=None, idxs_ca=None):
  dims = [0 for _ in range(len(idxs_resnet))]
  idx_map = [[0 for _ in range(5)] for _ in range(5)]
  for i, idx in enumerate(idxs_resnet):
    idx_map[idx[0]][idx[1]] = i
  for i, up_block in enumerate(unet.up_blocks):
      for j, module in enumerate(up_block.resnets):
          if [i, j] in idxs_resnet or (i, j) in idxs_resnet:
            # print('!!!! ', i, j, module.time_emb_proj.out_features, module)
            dims[idx_map[i][j]] += module.time_emb_proj.out_features
  for i, up_block in enumerate(unet.up_blocks):
    if hasattr(up_block, 'attentions'):
      for j, module in enumerate(up_block.attentions):
        if [i, j] in idxs_ca or (i, j) in idxs_ca:
          dims[idx_map[i][j]] += module.transformer_blocks[0].norm1.normalized_shape[0]
  return dims

def collect_feats_resnet(unet, idxs):
  feats = []
  layers = collect_layers_resnet(unet, idxs)
  for module in layers:
    # print(module.feats.shape)
    feats.append(module.feats)
  return feats

def collect_feats_ca(unet, idxs):
  feats = []
  layers = collect_layers_ca(unet, idxs)
  for module in layers:
    feats.append(module.feats)
  return feats

def set_feats(unet, feats, idxs):
  layers = collect_layers(unet, idxs)
  for i, module in enumerate(layers):
    module.feats = feats[i]