import torch
from tqdm import tqdm
import dl_utils
import torch.nn as nn
import time
import numpy as np
import cv2
import os

def check_updown(up_down):
    ud = up_down.split('-')[0]  # bic / avg / none
    ud_scale = int(up_down.split('-')[1])
    return ud, ud_scale


def eval_psnr_odi(loader, model):
    model.eval()
    metric_fn_psnr = dl_utils.calc_psnr

    val_res_psnr = dl_utils.Averager()

    pbar = tqdm(loader, leave=False, desc='eval_psnr')
    with torch.no_grad():
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.cuda()
            pred = model(x=batch['lr_img'], sample_coords=batch['coords_sample'], condition=batch['condition'],lonlat_hr=batch['lonlat_hr'],lonlat_lr=batch['lonlat_lr'], qmap=batch['qmap'])
            pred.clamp_(-1, 1)
            res_psnr = metric_fn_psnr(pred, batch, use_norm=False)
            val_res_psnr.add(res_psnr.item(), batch['lr_img'].shape[0])
            pbar.set_description('val {:.4f}'.format(val_res_psnr.item()))

    return val_res_psnr.item()

def test_both_ours(loader, model, log_fn, log_name, save_img=False, exp_folder='odisr'):
    model.eval()
    metric_fn_ssim = dl_utils.psnr_metric.cal_ssim
    metric_fn_psnr = dl_utils.calc_psnr

    ws = torch.from_numpy(np.load('./mw.npy')).view(1, 3, -1).unsqueeze(0).cuda()

    test_res_psnr = dl_utils.Averager()
    test_res_ssim = dl_utils.Averager()
    avg_time_encoder = dl_utils.Averager()
    avg_time_render = dl_utils.Averager()
    avg_time_all = dl_utils.Averager()
    pbar = tqdm(loader, leave=False, desc='test_both')
    id = 0
    total_time = 0.0
    with torch.no_grad():
        best_psnr = 0
        for batch in pbar:
            torch.cuda.empty_cache()
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.cuda()
            start_time = time.time()
            pred = model(x=batch['lr_img'], sample_coords=batch['coords_sample'], condition=batch['condition'],lonlat_hr=batch['lonlat_hr'],lonlat_lr=batch['lonlat_lr'], qmap=batch['qmap'])
            run_time = time.time() - start_time
            pred.clamp_(-1, 1)

            total_time += run_time
            res_psnr = metric_fn_psnr(pred, batch, ws=ws, if_ws=True)
            res_ssim = metric_fn_ssim(pred, batch['gt_sample'])
            if save_img:
                _, _, h, w = batch['lr_img'].shape
                scale = 1024 // h
                # print('scale = ', scale)
                save_pred = (pred + 1)/2 * 255
                save_folder = f'./vis_res/{exp_folder}/X{scale}'
                os.makedirs(save_folder, exist_ok=True)
                save_path = f'{save_folder}/{id}.jpg'
                save_pred = save_pred.squeeze(0)
                save_pred = save_pred.reshape(3, 1024, 2048)
                save_pred = save_pred.cpu().numpy()
                save_pred = save_pred.astype(np.uint8)
                save_pred = save_pred.transpose(1, 2, 0)
                save_pred = save_pred[:, :, [2, 1, 0]]
                cv2.imwrite(save_path, save_pred)

            if res_psnr >= best_psnr:
                best_psnr = res_psnr
                # print('best psnr = ', best_psnr)
                log_fn(
                    f'test_img: {id}, best psnr: {res_psnr.item()}, ssim: {res_ssim.item()}, time: {run_time}s',
                    filename=log_name)
                

            else:
                log_fn(
                    f'test_img: {id}, psnr: {res_psnr.item()}, ssim: {res_ssim.item()}, time: {run_time}s',
                    filename=log_name)
            test_res_psnr.add(res_psnr.item(), batch['gt_sample'].shape[0])
            test_res_ssim.add(res_ssim.item(), batch['gt_sample'].shape[0])
            avg_time_all.add(run_time, batch['gt_sample'].shape[0])
            id += 1
            pbar.set_description('img:{}, psnr: {:.4f}, ssim: {:.4f}'.format(id - 1, res_psnr.item(), res_ssim.item()))

    log_fn(f'total time = {total_time}s', filename=log_name)
    return test_res_psnr.item(), test_res_ssim.item(), [avg_time_encoder.item(), avg_time_render.item(),
                                                      avg_time_all.item()]

def single_img_sr(lr_img, model, h, w, gt=None, up_down=None, flip=None):
    model.eval()
    ud = 'none'
    ud_scale = 1
    if up_down is not None:
        ud, ud_scale = check_updown(up_down)
    with torch.no_grad():
        if flip is not None:
            pred, run_time = model.inference(lr_img, h=h, w=w, flip_conf=flip)
            pred.clamp_(-1, 1)
        else:
            if ud == 'none':
                pred, run_time = model.inference(lr_img, h=h, w=w)
                pred.clamp_(-1, 1)
            elif ud == 'bic':
                pred, run_time = model.inference(lr_img, h=h * ud_scale, w=w * ud_scale)

                pred = dl_utils.resize_img(pred, (h, w)).cuda()
                pred.clamp_(-1, 1)
            elif ud == 'avg':
                pred, run_time = model.inference(lr_img, h=h * ud_scale, w=w * ud_scale)
                m = nn.AdaptiveAvgPool2d((h, w))
                pred = m(pred)
                pred.clamp_(-1, 1)

            else:
                RuntimeError('updown fault')

        if gt is not None:
            metric_fn_psnr = dl_utils.calc_psnr
            metric_fn_ssim = dl_utils.calc_ssim
            res_psnr = metric_fn_psnr(pred, gt)
            res_ssim = metric_fn_ssim(pred, gt)
            return pred, res_psnr, res_ssim, run_time
        else:
            return pred, None, None, run_time