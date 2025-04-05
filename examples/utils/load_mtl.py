
import os
import torch
import torch.distributed as dist
from torch import inf
import errno

from PIL import Image
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from models.lora import map_old_state_dict_weights

def load_checkpoint(config, model, backbone=False, quiet=False):
    # resume_path = config.MODEL.RESUME if not backbone else config.MODEL.RESUME_BACKBONE
    resume_path = config.MODEL.PATH
    # logger.info(
    #     f"==============> Resuming form {resume_path}....................")
    if resume_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            resume_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(resume_path, map_location='cpu')

    mtlora = config.MODEL.MTLORA
    mtlora_enabled = mtlora.ENABLED

    # skip_decoder = config.TRAIN.SKIP_DECODER_CKPT
    # model_state = {k: v for k, v in checkpoint["model"].items(
    # ) if not k.startswith("decoders")} if skip_decoder else checkpoint["model"]
    
    model_state = checkpoint["model"]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in model_state.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del model_state[k]

    if config.MODEL.UPDATE_RELATIVE_POSITION:
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in model_state.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del model_state[k]

        # delete relative_coords_table since we always re-init it
        relative_position_index_keys = [
            k for k in model_state.keys() if "relative_coords_table" in k]
        for k in relative_position_index_keys:
            del model_state[k]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in model_state.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = model_state[k]
            relative_position_bias_table_current = model.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    # bicubic interpolate relative_position_bias_table if not match
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                        mode='bicubic')
                    model_state[k] = relative_position_bias_table_pretrained_resized.view(
                        nH2, L2).permute(1, 0)

        # bicubic interpolate absolute_pos_embed if not match
        absolute_pos_embed_keys = [
            k for k in model_state.keys() if "absolute_pos_embed" in k]
        for k in absolute_pos_embed_keys:
            # dpe
            absolute_pos_embed_pretrained = model_state[k]
            absolute_pos_embed_current = model.model_state()[k]
            _, L1, C1 = absolute_pos_embed_pretrained.size()
            _, L2, C2 = absolute_pos_embed_current.size()
            if C1 != C1:
                print(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                        -1, S1, S1, C1)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                        0, 3, 1, 2)
                    absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                        absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(
                        0, 2, 3, 1)
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(
                        1, 2)
                    model_state[k] = absolute_pos_embed_pretrained_resized

    if mtlora_enabled:
        mapping = {}
        trainable_layers = []
        if mtlora.QKV_ENABLED:
            trainable_layers.extend(["attn.qkv.weight", "attn.qkv.bias"])
        if mtlora.PROJ_ENABLED:
            trainable_layers.extend(["attn.proj.weight", "attn.proj.bias"])
        if mtlora.FC1_ENABLED:
            trainable_layers.extend(["mlp.fc1.weight", "mlp.fc1.bias"])
        if mtlora.FC2_ENABLED:
            trainable_layers.extend(["mlp.fc2.weight", "mlp.fc2.bias"])
        if mtlora.DOWNSAMPLER_ENABLED:
            trainable_layers.extend(["downsample.reduction.weight"])

        for k, v in model_state.items():
            last_three = ".".join(k.split(".")[-3:])
            prefix = ".".join(k.split(".")[:-3])
            if last_three in trainable_layers:
                weight_bias = last_three.split(".")[-1]
                layer_name = ".".join(last_three.split(".")[:-1])
                mapping[f"{prefix}.{layer_name}.{weight_bias}"] = f"{prefix}.{layer_name}.linear.{weight_bias}"
        if not len(mapping):
            print("No keys needs to be mapped for LoRA")
        model_state = map_old_state_dict_weights(
            model_state, mapping, "", config.MODEL.MTLORA.SPLIT_QKV)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if not quiet:
        if len(missing) > 0:
            print("=============Missing Keys==============")
            for k in missing:
                print(k)
        if len(unexpected) > 0:
            print("=============Unexpected Keys==============")
            for k in unexpected:
                print(k)
    max_accuracy = 0.0
    # if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint and not skip_decoder:
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #     lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    #     config.defrost()
    #     config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
    #     config.freeze()
    #     if 'scaler' in checkpoint:
    #         loss_scaler.load_state_dict(checkpoint['scaler'])
    #     logger.info(
    #         f"=> loaded successfully '{resume_path}' (epoch {checkpoint['epoch']})")
    #     if 'max_accuracy' in checkpoint:
    #         max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy