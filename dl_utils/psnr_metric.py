import torch
import numpy as np
import cv2
import math
from torchvision import transforms

from basicsr.metrics.metric_util import reorder_image, to_y_channel

def calculate_psnr_ws(img, img2, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate WS-PSNR.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: WS-PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    img_w = compute_map_ws(img)

    mse = np.mean(np.multiply((img - img2)**2, img_w))/np.mean(img_w)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)

def compute_map_ws(img):
    """calculate weights for the sphere, the function provide weighting map for a given video
        :img(HWC)    the input original video
    """
    equ = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for i in range(0,equ.shape[0]):
        for j in range(0,equ.shape[1]):
            for k in range(0,equ.shape[2]):
                equ[i, j, k] = genERP(i,equ.shape[0])
    return equ

def calc_psnr(sr, batch, ws=1, scale=1, rgb_range=1., use_norm=True, if_ws=False):
    """
    input: [-1,1] not normalize
    """
    hr = batch['gt_sample']
    if use_norm is True:
        sr = (sr + 1) / 2
        hr = (hr + 1) / 2
    diff = (sr - hr) / rgb_range
    valid = diff
    if if_ws:
        mse = (valid.pow(2) * ws).sum() / ws.sum()
    else:
        ws = torch.cos(batch['lonlat_hr'][:,:,0:1] * torch.pi / 2).permute(0,2,1)
        mse = (valid.pow(2) * ws).sum() / ws.sum()

    return -10 * torch.log10(mse)


def cal_ssim(img1, img2):
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    img1 = img1.cpu().view(3,1024,2048).numpy() * 255
    img2 = img2.cpu().view(3,1024,2048).numpy() * 255
    img1 = img1.astype(np.int64)
    img2 = img2.astype(np.int64)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # print(img1.shape)
    img1 = img1.transpose(1,2,0)
    img2 = img2.transpose(1,2,0)
    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ws_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()
    

def _ws_ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    equ = np.zeros((ssim_map.shape[0], ssim_map.shape[1]))

    for i in range(0,equ.shape[0]):
        for j in range(0,equ.shape[1]):
                equ[i, j] = genERP(i,equ.shape[0])

    return np.multiply(ssim_map, equ).mean()/equ.mean()

def genERP(j,N):
    val = math.pi/N
    w = math.cos((j - (N/2) + 0.5) * val)
    return w