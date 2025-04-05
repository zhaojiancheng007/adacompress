import torch.nn.functional as F
def pad(x, p=2**6):  # 默认填充到 64 的倍数，可修改为 512
    """
    Pads input tensor on the bottom and right to be a multiple of p.
    Uses replicate padding.

    Args:
        x (Tensor): [B, C, H, W]
        p (int): multiple (default 64)

    Returns:
        padded_x (Tensor): padded tensor
        padding (Tuple[int, int]): (pad_right, pad_bottom)
    """
    _, _, h, w = x.size()
    pad_h = (p - h % p) % p
    pad_w = (p - w % p) % p
    padded_x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return padded_x, (pad_w, pad_h)

def remove_padding_tensor(x_tensor, padding):
    """
    裁剪单个张量，移除右侧和下方 padding
    Args:
        x_tensor: [B, C, H_pad, W_pad]
        padding: (pad_right, pad_bottom)
    Returns:
        [B, C, H, W]
    """
    pad_right, pad_bottom = padding
    if pad_bottom > 0:
        x_tensor = x_tensor[:, :, :-pad_bottom, :]
    if pad_right > 0:
        x_tensor = x_tensor[:, :, :, :-pad_right]
    return x_tensor

def remove_padding_dict(x_dict, padding):
    """
    裁剪字典中的特定 key 图像，返回裁剪后的字典
    Args:
        x_dict: {key: Tensor}
        padding: (pad_right, pad_bottom)
    Returns:
        新字典
    """
    keys_to_crop = ['x_hat', 'semseg', 'human_parts', 'sal', 'normals']
    out_cropped = {}
    for key, value in x_dict.items():
        if key in keys_to_crop:
            out_cropped[key] = remove_padding_tensor(value, padding)
        else:
            out_cropped[key] = value  # 其他 key 保留原值
    return out_cropped
