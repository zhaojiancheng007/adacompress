import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        id = i
        for j in range(7):
            str_id = np.binary_repr(id, width=8)
            r ^= (np.uint8(str_id[-1]) << (7-j))
            g ^= (np.uint8(str_id[-2]) << (7-j))
            b ^= (np.uint8(str_id[-3]) << (7-j))
            id >>= 3
        cmap[i] = [r, g, b]
    return cmap

def vis_semseg(_semseg):
    new_cmap = labelcolormap(21)  # PASCAL CONTEXT有21类
    return new_cmap[_semseg]

def vis_parts(_semseg):
    new_cmap = labelcolormap(7)   # 假设你的人体部件只有7类
    return new_cmap[_semseg]

def tens2image(tens, transpose=False):
    """Converts tensor with 2 or 3 dimensions to numpy array"""
    im = tens.cpu().detach().numpy()

    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    elif im.shape[-1] == 1:
        im = np.squeeze(im)
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    if transpose:
        if im.ndim == 3:
            im = im.transpose((1, 2, 0))
    return im


def normalize(arr, t_min=0, t_max=255):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = arr.max() - arr.min()
    for i in arr:
        temp = (((i - arr.min())*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    res = np.array(norm_arr)
    return res


def compute_colored_spectrum(image_tensor, colormap=cv2.COLORMAP_OCEAN):
    """
    计算图像的频谱并应用指定的颜色映射。

    参数:
        image_tensor (torch.Tensor 或 np.ndarray): 输入图像，形状为 [C, H, W] 或 [H, W]。
        colormap (int): OpenCV 的颜色映射，默认为 cv2.COLORMAP_OCEAN。

    返回:
        np.ndarray: 彩色频谱图，形状为 [H, W, 3]。
    """
    # 将 PyTorch 张量转换为 NumPy 数组
    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.detach().cpu().numpy()
    else:
        image_np = image_tensor

    # 如果是彩色图像，转换为灰度图像
    if image_np.ndim == 3 and image_np.shape[0] == 3:
        image_np = 0.2989 * image_np[0] + 0.5870 * image_np[1] + 0.1140 * image_np[2]

    # 计算频谱
    f = np.fft.fft2(image_np)
    fshift = np.fft.fftshift(f)
    spectrum = np.abs(fshift)
    spectrum_log = np.log1p(spectrum)  # 使用对数尺度以增强可视化

    # 归一化到 0-255
    spectrum_norm = cv2.normalize(spectrum_log, None, 0, 255, cv2.NORM_MINMAX)
    spectrum_uint8 = np.uint8(spectrum_norm)

    # 应用颜色映射
    spectrum_color = cv2.applyColorMap(spectrum_uint8, colormap)

    return spectrum_color

def save_spectrum_image(img_tensor, save_path, title=None):
    """
    输入 shape=(3,H,W) 图像张量，保存频谱图，默认使用 viridis（蓝绿色）颜色映射，并带颜色条
    """
    # 1. 转换为 numpy 并转灰度
    from matplotlib.colors import Normalize
    from matplotlib import cm

    if isinstance(img_tensor, torch.Tensor):
        img_np = img_tensor.detach().cpu().numpy()
    else:
        img_np = img_tensor

    if img_np.ndim == 3 and img_np.shape[0] == 3:
        # RGB to gray
        img_np = 0.2989 * img_np[0] + 0.5870 * img_np[1] + 0.1140 * img_np[2]

    # 2. 计算频谱
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    spectrum = np.log1p(np.abs(fshift))

    # 3. 使用 viridis colormap 映射
    norm = Normalize(vmin=np.min(spectrum), vmax=np.max(spectrum))
    colormap = cm.get_cmap('viridis')
    spectrum_color = colormap(norm(spectrum))  # (H,W,4) 带透明度
    spectrum_color = (spectrum_color[:, :, :3] * 255).astype(np.uint8)

    # 4. 保存
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(spectrum_color)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)

    # 添加颜色条
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, shrink=0.7, fraction=0.046)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Log Magnitude", fontsize=10)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()

def save_imgs_mtl(batch_imgs, recon_semseg, recon_human, recon_sal, recon_normal,
                  batch_labels, batch_predictions, path, id):
    import torchvision
    
    recon_images = {
        "recon_semseg": tens2image(recon_semseg, transpose=True),
        "recon_human": tens2image(recon_human, transpose=True),
        "recon_sal": tens2image(recon_sal, transpose=True),
        "recon_normal": tens2image(recon_normal, transpose=True),
    }

    imgs = tens2image(batch_imgs, transpose=True)
    labels = {task: tens2image(label, transpose=True)
              for task, label in batch_labels.items()}
    predictions = {task: tens2image(prediction)
                   for task, prediction in batch_predictions.items()}

    Image.fromarray(normalize(imgs, 0, 255).astype(
        np.uint8)).save(f'{path}/{id}_img.png')
    
    for key, img in recon_images.items():
        Image.fromarray(normalize(img, 0, 255).astype(np.uint8)).save(f"{path}/{id}_{key}.png")
    
    save_spectrum_image(imgs.transpose(2, 0, 1), f"{path}/{id}_input_spectrum.png")
    for key, img in recon_images.items():
        save_spectrum_image(img.transpose(2, 0, 1), f"{path}/{id}_{key}_spectrum.png")
    
    for task in labels.keys():
        if task in ["semseg"]:
            print(np.sum(labels[task] != 255))
            labels[task] = labels[task] != 255
            predictions[task] = predictions[task] != 225
            batch_imgs = 255*(batch_imgs-torch.min(batch_imgs)) / \
                (torch.max(batch_imgs)-torch.min(batch_imgs))
            # print('#'*10,batch_imgs[0].shape, batch_predictions[task][0].shape)
            semseg = torchvision.utils.draw_segmentation_masks(batch_imgs.cpu().detach().to(torch.uint8),
                                                               batch_predictions[task].to(torch.bool), colors="blue", alpha=0.5)
            Image.fromarray(semseg.numpy().transpose((1, 2, 0))
                            ).save(f'{path}/{id}_{task}_pred.png')
            semseg = torchvision.utils.draw_segmentation_masks(batch_imgs.cpu().detach().to(torch.uint8),
                                                               batch_labels[task].to(torch.bool), colors="blue", alpha=0.5)
            Image.fromarray(semseg.numpy().transpose((1, 2, 0))
                            ).save(f'{path}/{id}_{task}_gt.png')
        
        elif task == "human_parts":
            # 添加颜色映射：7 个 parts
            cmap = labelcolormap(7)

            gt = labels[task].astype(np.uint8)
            pred = predictions[task].astype(np.uint8)

            gt_color = cmap[np.clip(gt, 0, 6)]  # 防止值超界
            pred_color = cmap[np.clip(pred, 0, 6)]

            Image.fromarray(gt_color).save(f'{path}/{id}_{task}_gt.png')
            Image.fromarray(pred_color).save(f'{path}/{id}_{task}_pred.png')
        
        else:
            labels[task] = normalize(labels[task], 0, 255)
            predictions[task] = normalize(predictions[task], 0, 255)
            
            Image.fromarray(labels[task].astype(np.uint8)).save(
                f'{path}/{id}_{task}_gt.png')
            Image.fromarray(predictions[task].astype(np.uint8)).save(
                f'{path}/{id}_{task}_pred.png')
