import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
import datasets
import models
import dl_utils
from tensorboardX import SummaryWriter
from test import test_both_ours


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='save/Test/')
    parser.add_argument('--test_config', default='configs/test-configs/test_ODI-SEG-SR.yaml')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.test_config, 'r') as f:
        test_config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    ckpt_path = test_config['checkpoint']
    exp_name = test_config['exp_name']
    is_save = test_config['save_img']
    test_name = args.test_config.split('/')[-1].split('.')[-2] + '-' + str(exp_name)
    save_dir = os.path.join(args.exp_folder + test_name)
    log, _ = dl_utils.set_save_path(save_dir, writer=False)
    os.makedirs(save_dir, exist_ok=True)
    print(ckpt_path)

    log_name = '_log.txt'
    test_spec = test_config['test_dataset']
    sv_file = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model = models.make(sv_file['model'], load_sd=True).cuda()
    total = sum([param.nelement() for param in model.parameters()])
    log('ckpt path: ' + ckpt_path, filename=log_name)
    log("Number of parameter: %.4fM" % (total/1e6), filename=log_name)
    log('dataset path: ' + test_spec['dataset']['args']['root_path'], filename=log_name)
    for xn in test_spec['eval_n']:
        dataset = datasets.make(test_spec['dataset'])
        dataset = datasets.make(test_spec['wrapper'], args={'dataset': dataset, 'xn': xn})
        loader = DataLoader(dataset, batch_size=test_spec['batch_size'],
                            num_workers=8, pin_memory=True)

        test_psnr, test_ssim, test_run_time = test_both_ours(loader, model, log, log_name, save_img=is_save)

        log('test avg: psnr={:.4f}'.format(test_psnr), filename=log_name)
        log('test avg: ssim={:.4f}'.format(test_ssim), filename=log_name)
        log(f'test x{xn} avg encoder time: {test_run_time[0]}s', filename=log_name)
        log(f'test x{xn} avg render time: {test_run_time[1]}s', filename=log_name)
        log(f'test x{xn} avg all time: {test_run_time[2]}s', filename=log_name)
        log(f'\n\n\n', filename=log_name)
