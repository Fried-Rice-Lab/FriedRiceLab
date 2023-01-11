# --------------------------------------------------------------------------------
# Basic model for image restoration with infer, interpret and analysis functions.
# Supports pixel value between [0, 2 ** Bit - 1].
#
# Implemented by Jinpeng Shi (https://github.com/jinpeng-s)
# --------------------------------------------------------------------------------
from os import path as osp

import cv2
import imageio
import numpy as np
import torch
from basicsr.metrics import calculate_metric  # noqa
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from tqdm import tqdm


@MODEL_REGISTRY.register()
class IRModel(SRModel):
    r"""Basic model for image restoration.
    """

    def __init__(self, opt) -> None:
        super(IRModel, self).__init__(opt)

        self.bit = self.opt['bit']

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += "\n[{}]   [{:<16} @ {}]".format(
                "{}".format(metric),
                "Current: {}".format(round(value, 4)),
                "iter {}".format(current_iter)
            )
            if hasattr(self, 'best_metric_results'):
                log_str += "   [{:<13} @ {}]".format(
                    "Best: {}".format(round(self.best_metric_results[dataset_name][metric]['val'], 4)),
                    "iter {}".format(self.best_metric_results[dataset_name][metric]['iter'])
                )
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def test_selfensemble(self):
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)  # noqa
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)  # noqa

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        self_ensemble = self.opt['val'].get('self_ensemble', False)

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
            self.test_selfensemble() if self_ensemble else self.test()

            visuals = self.get_current_visuals()
            if self.bit == 0:
                sr_img = tensor2img([visuals['result']])
                metric_data['img'] = sr_img
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']])
                    metric_data['img2'] = gt_img
                    del self.gt
            else:
                temp_img_path = osp.join(self.opt['path']['visualization'], 'temp.png')
                sr_img = visuals['result'].squeeze(0).detach().cpu(). \
                    clamp(0, 2 ** self.bit - 1.).round().numpy().transpose(1, 2, 0)
                imageio.imwrite(temp_img_path, sr_img.astype(np.uint8))
                sr_img = tensor2img(img2tensor(cv2.imread(temp_img_path) / (2 ** self.bit - 1.)))  # noqa
                metric_data['img'] = sr_img
                if 'gt' in visuals:
                    gt_img = visuals['gt'].squeeze(0).detach().cpu(). \
                        clamp(0, 2 ** self.bit - 1.).round().numpy().transpose(1, 2, 0)
                    imageio.imwrite(temp_img_path, gt_img.astype(np.uint8))
                    gt_img = tensor2img(img2tensor(cv2.imread(temp_img_path) / (2 ** self.bit - 1.)))  # noqa
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
            if self.bit == 0:
                sr_img = tensor2img([visuals['result']])
            else:
                temp_img_path = osp.join(self.opt['path']['visualization'], 'temp.png')
                sr_img = visuals['result'].squeeze(0).detach().cpu(). \
                    clamp(0, 2 ** self.bit - 1.).round().numpy().transpose(1, 2, 0)
                imageio.imwrite(temp_img_path, sr_img.astype(np.uint8))
                sr_img = tensor2img(img2tensor(cv2.imread(temp_img_path) / (2 ** self.bit - 1.)))  # noqa

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # save inference results
            save_img_path = osp.join(self.opt['path']['visualization'], 'inference', dataset_name, 'BI',
                                     f"x{self.opt['scale']}", f'{img_name}.png')
            imwrite(sr_img, save_img_path)

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
