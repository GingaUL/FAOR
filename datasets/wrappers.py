import random
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets import register
from dl_utils import make_coord


def resize_fn(img, size, is_img=True):
    if is_img:
        return transforms.ToTensor()(
            transforms.Resize(size, InterpolationMode.BICUBIC)(
                transforms.ToPILImage()(img)))
    else:
        img = F.interpolate(img.unsqueeze(0), size=size, mode='bilinear')
        return img[0]


@register('ours-sample-train')
class ours_sample(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=True, norm=True, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.norm = norm
        self.ws = torch.from_numpy(np.load('../mw.npy')).float()
        self.coord = make_coord((1024,2048), flatten=False)  #[H, W, 2]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, map = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)
        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[-2] - w_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_map = map[:, x0:x0 + w_hr, y0: y0 + w_hr]
        lonlat_hr = self.coord[x0: x0 + w_hr, y0: y0 + w_hr, :].permute(2,0,1)      #[2,W,H]  
        lonlat_lr = resize_fn(lonlat_hr, w_lr, is_img=False)
        crop_lr = resize_fn(crop_hr, w_lr)  # img_lr_s
        crop_map = resize_fn(crop_map, w_lr)
        crop_condition = self.ws[:1, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_condition = resize_fn(crop_condition, w_lr, is_img=False)
        

        if self.augment:
            vflip = random.random() < 0.5
            
            def augment(x):
                if vflip:
                    x = x.flip(-1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            crop_condition = augment(crop_condition)
        lonlat_hr = lonlat_hr.contiguous().view(2, -1).permute(1, 0)
        hr_coord = make_coord((w_hr, w_hr), flatten=True)  # [wh, 2]
        crop_hr = crop_hr.contiguous().view(3, -1).permute(1, 0)
        
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            crop_hr = crop_hr[sample_lst]
            lonlat_hr = lonlat_hr[sample_lst]
        if self.norm:
            crop_lr = (crop_lr - 0.5) / 0.5
            crop_hr = (crop_hr - 0.5) / 0.5
        sample_batch = {
            'lr_img': crop_lr,
            'coords_sample': hr_coord,
            'gt_sample': crop_hr.permute(1, 0),
            'condition': crop_condition,
            'lonlat_hr': lonlat_hr,
            'lonlat_lr': lonlat_lr,
            'qmap': crop_map
        }

        return sample_batch

@register('ours-test-xn')  # Downsample HR-ODI to LR-ODI then SR to HR-ODI
class our_patch(Dataset):
    def __init__(self, dataset, xn=4, inp_size=None, norm=True, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.norm = norm
        self.xn = xn
        self.ws = torch.from_numpy(np.load('mw.npy')).float()
        self.lontlat_hr = make_coord((1024, 2048), flatten=True)
        self.lontlat_lr = make_coord((1024 // xn, 2048 // xn), flatten=False).permute(2,0,1)  
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, map = self.dataset[idx]
        crop_hr, crop_map = img, map
        crop_lr = resize_fn(crop_hr, (1024 // self.xn, 2048 // self.xn))
        crop_map = resize_fn(crop_map, (1024 // self.xn, 2048 // self.xn))
        crop_condition = self.ws[:1, :, :]
        crop_condition = resize_fn(crop_condition, crop_lr.shape[-2:], is_img=False)
        crop_hr = crop_hr.contiguous().view(3, -1)

        if self.norm:
            crop_lr = (crop_lr - 0.5) / 0.5
            crop_hr = (crop_hr - 0.5) / 0.5
        return {
            'lr_img': crop_lr,
            'gt_sample': crop_hr,
            'coords_sample': self.lontlat_hr,
            'lonlat_hr': self.lontlat_hr,
            'lonlat_lr': self.lontlat_lr,
            'condition': crop_condition,
            'qmap': crop_map,
        }

@register('ours-demo-xn')  # SR LR-ODI from h, w to h*xn, w*xn HR-ODI
class test_patch(Dataset):
    def __init__(self, dataset, xn=4, inp_size=None, norm=True, augment=False, h=1024, w=2048):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.norm = norm
        self.xn = xn
        self.ws = torch.cos(make_coord([h]).unsqueeze(1).repeat([1, w, 1]).permute(2,0,1) * math.pi / 2)  # 1, h, w
        self.lontlat_hr = make_coord((h*xn, w*xn), flatten=True)
        self.lontlat_lr = make_coord((h, w), flatten=False).permute(2,0,1)  
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, map = self.dataset[idx]
        crop_hr, crop_map = img, map
        crop_lr = resize_fn(crop_hr, (1024 // self.xn, 2048 // self.xn))
        crop_map = resize_fn(crop_map, (1024 // self.xn, 2048 // self.xn))
        crop_condition = self.ws
        crop_condition = resize_fn(crop_condition, crop_lr.shape[-2:], is_img=False)
        crop_hr = crop_hr.contiguous().view(3, -1)

        if self.norm:
            crop_lr = (crop_lr - 0.5) / 0.5
            crop_hr = (crop_hr - 0.5) / 0.5
        return {
            'lr_img': crop_lr,
            'gt_sample': crop_hr,
            'coords_sample': self.lontlat_hr,
            'lonlat_hr': self.lontlat_hr,
            'lonlat_lr': self.lontlat_lr,
            'condition': crop_condition,
            'qmap': crop_map,
        }