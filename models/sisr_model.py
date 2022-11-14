# --------------------------------------------------------------------------------
# Enhanced Basic SR Model with inference, interpretation and analysis functions.
# Supports pixel value between [0, 1] or [0, 255].
#
# Implemented by Jinpeng Shi (https://github.com/jinpeng-s)
# --------------------------------------------------------------------------------
import os
from os import path as osp

import imageio
import numpy as np
import torch
from basicsr.metrics import calculate_metric  # noqa
from basicsr.models.sr_model import SRModel
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from tqdm import tqdm


@MODEL_REGISTRY.register()
class SISRModel(SRModel):
    r"""Base SR model for single image super-resolution.
        Please use with SISRDataset.
    """

    def __init__(self, opt) -> None:
        super(SISRModel, self).__init__(opt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}  # noqa

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'train', img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], 'test', dataset_name, 'BI',
                                                 f"x{self.opt['scale']}", f'{img_name}_{self.opt["name"]}.png')

                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], 'test', dataset_name, 'BI',
                                                 f"x{self.opt['scale']}", f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)  # noqa
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)  # noqa
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def nondist_inference(self, dataloader) -> None:
        dataset_name = dataloader.dataset.opt['name']

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])  # !

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # save inference results
            save_img_path = osp.join(self.opt['path']['visualization'], 'inference', dataset_name, 'BI',
                                     f"x{self.opt['scale']}", f'{img_name}.png')
            os.makedirs(os.path.abspath(os.path.dirname(save_img_path)), exist_ok=True)
            imwrite(sr_img, save_img_path)  # !

    def nondist_analysis(self, dataloader) -> tuple:
        self.net_g.eval()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        time_list = list()
        for idx, val_data in enumerate(dataloader):
            self.lq = val_data['lq'].to(self.device)  # noqa
            start.record()
            with torch.no_grad():
                self.net_g(self.lq)  # noqa
            end.record()
            torch.cuda.synchronize()
            time_list.append(start.elapsed_time(end))

        ave_time = sum(time_list) / len(time_list)
        gpu_mem = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2

        return ave_time, gpu_mem


@MODEL_REGISTRY.register()
class SISRModel8Bit(SRModel):
    r"""Base SR model for single image super-resolution.
        Please use with SISRDataset8Bit.
    """

    def __init__(self, opt) -> None:
        super(SISRModel8Bit, self).__init__(opt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img) -> None:
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}  # noqa

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # sr_img = tensor2img([visuals['result']])
            sr_img = visuals['result'].squeeze(0).detach().cpu().clamp(0, 255).round().numpy().transpose(1, 2, 0)
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                # gt_img = tensor2img([visuals['gt']])
                gt_img = visuals['gt'].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'train', img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], 'test', dataset_name, 'BI',
                                                 f"x{self.opt['scale']}", f'{img_name}_{self.opt["name"]}.png')

                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], 'test', dataset_name, 'BI',
                                                 f"x{self.opt['scale']}", f'{img_name}.png')

                # imwrite(sr_img, save_img_path)
                os.makedirs(os.path.abspath(os.path.dirname(save_img_path)), exist_ok=True)
                imageio.imwrite(save_img_path, sr_img.astype(np.uint8))

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)  # noqa
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)  # noqa
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def nondist_inference(self, dataloader) -> None:
        dataset_name = dataloader.dataset.opt['name']

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = visuals['result'].squeeze(0).detach().cpu().clamp(0, 255).round().numpy().transpose(1, 2, 0)

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # save inference results
            save_img_path = osp.join(self.opt['path']['visualization'], 'inference', dataset_name, 'BI',
                                     f"x{self.opt['scale']}", f'{img_name}.png')
            os.makedirs(os.path.abspath(os.path.dirname(save_img_path)), exist_ok=True)
            imageio.imwrite(save_img_path, sr_img.astype(np.uint8))

    def nondist_analysis(self, dataloader) -> tuple:
        self.net_g.eval()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        time_list = list()
        for idx, val_data in enumerate(dataloader):
            self.lq = val_data['lq'].to(self.device)  # noqa
            start.record()
            with torch.no_grad():
                self.net_g(self.lq)  # noqa
            end.record()
            torch.cuda.synchronize()
            time_list.append(start.elapsed_time(end))

        ave_time = sum(time_list) / len(time_list)
        gpu_mem = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2

        return ave_time, gpu_mem
