import os.path
import logging
import torch
import argparse
import json

from pprint import pprint
from tools.analyse_tool.model_summary import get_model_activation, get_model_flops
from tools.analyse_tool import utils_logger, utils_image as util


def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == -1:
        # IMDN baseline
        name, data_range = f"{model_id:02}_IMDN_baseline", 1.0
        from models.imdn_baseline import IMDN
        model_path = os.path.join("model_zoo", "imdn_baseline.pth")
        model = IMDN(in_nc=3, out_nc=3, nc=64, nb=8, upscale=4)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 0:
        # RFDN baseline, AIM 2020 Efficient SR Challenge winner
        from models.rfdn_baseline.RFDN import RFDN
        name, data_range = f"{model_id:02}_RFDN_baseline", 255.0
        model_path = os.path.join('model_zoo', 'rfdn_baseline.pth')
        model = RFDN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 1:
        # NKU-ESR Team
        from models.team01_efdn import EFDN
        name, data_range = f"{model_id:02}_EFDN", 1.0
        model_path = os.path.join('model_zoo', 'team01_efdn.pth')
        model = EFDN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 2:
        # Virtual_Reality Team
        from models.team02_nlffc.NLFFC import Netw
        name, data_range = f"{model_id:02}_NLFFC", 255.0
        model_path = os.path.join('model_zoo', 'team02_nlffc.pth')
        model = Netw()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 3:
        # NJU_Jet Team
        from models.team03_fmen import FMEN
        name, data_range = f"{model_id:02}_FMEN", 255.0
        model_path = os.path.join('model_zoo', 'team03_fmen.pth')
        model = FMEN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 4:
        # ByteESR Team
        from models.team04_rlfn import RLFN_cut
        name, data_range = f"{model_id:02}_RLFN", 255.0
        model_path = os.path.join('model_zoo', 'team04_rlfn.pth')
        model = RLFN_cut(in_nc=3, out_nc=3)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 5:
        # NEESR Team
        from models.team05_efdn.plainsr import PLAINRFDN
        name, data_range = f"{model_id:02}_EFDN", 255.0
        model_path = os.path.join('model_zoo', 'team05_efdn.pt')
        model = PLAINRFDN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 6:
        # TieGuoDun Team
        from models.team06_v1 import v1
        name, data_range = f"{model_id:02}_V1", 1.0
        model_path = os.path.join('model_zoo', 'team06_v1.pth')
        model = v1(in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 7:
        # Express Team
        pass
    elif model_id == 8:
        # NTU607QCO-ESR Team
        from models.team08_sfdn import RFDN
        name, data_range = f"{model_id:02}_RFDN", 1.0
        model_path = os.path.join('model_zoo', 'team08_sfdn.pt')
        model = RFDN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 9:
        # ZLZ Team
        from models.team09_imdtn.architecture import IMDTN
        name, data_range = f"{model_id:02}_IMDTN", 1.0
        model_path = os.path.join('model_zoo', 'team09_imdtn.pth')
        model = IMDTN(upscale=4)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 10:
        # Super Team
        from models.team10_repafdn.repafdn import RePAFDN
        name, data_range = f"{model_id:02}_RePAFDN", 1.0
        model_path = os.path.join('model_zoo', 'team10_repafdn.pth')
        model = RePAFDN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 11:
        # Drinktea Team
        from models.team11_aaln.aaln import AALN
        name, data_range = f"{model_id:02}_AALN", 255.0
        model_path = os.path.join('model_zoo', 'team11_aaln.pt')
        model = AALN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 12:
        # mju_mnu Team
        from models.team12_hnct.HNCT import HNCT
        name, data_range = f"{model_id:02}_HNCT", 1.0
        model_path = os.path.join('model_zoo', 'team12_hnct.pt')
        model = HNCT()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 13:
        # whu_sigma Team
        from models.team13_rfdn_dilated.RFDN_dilated import RFDN_dilated
        name, data_range = f"{model_id:02}_RFDN_Dilated", 1.0
        model_path = os.path.join('model_zoo', 'team13_rfdn_dilated.pth')
        model = RFDN_dilated()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 14:
        # HiImageTeam
        from models.team14_arfdn.ARFDN import ARFDN
        name, data_range = f"{model_id:02}_ARFDN", 1.0
        model_path = os.path.join('model_zoo', 'team14_arfdn.pth')
        model = ARFDN()
        state_dict = torch.load(model_path)
        new_state_dict = dict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        model.load_state_dict(new_state_dict, strict=True)
    elif model_id == 15:
        # NJUST_RESTORARION Team
        from models.team15_afdn.AFDN import AFDN
        name, data_range = f"{model_id:02}_AFDN", 255.0
        model_path = os.path.join('model_zoo', 'team15_afdn.pt')
        model = AFDN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 16:
        # GDUT_SR Team
        from models.team16_prrn.PRRN import PRRN
        name, data_range = f"{model_id:02}_PRRN", 1.0
        model_path = os.path.join('model_zoo', 'team16_prrn.pth')
        model = PRRN(scale=4)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 17:
        # NJU_MCG Team
        from models.team17_fden.FDEN import FDEN
        name, data_range = f"{model_id:02}_FDEN", 255.0
        model_path = os.path.join('model_zoo', 'team17_fden.pth')
        model = FDEN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 18:
        # XPixel Team
        from models.team18_bsrn import BSRN
        name, data_range = f"{model_id:02}_RFDNFINALB5", 1.0
        model_path = os.path.join('model_zoo', 'team18_bsrn.pth')
        model = BSRN(num_in_ch=3, num_feat=48, num_block=5, num_out_ch=3, upscale=4,
                            conv='BSConvU', upsampler='pixelshuffledirect')
        model.load_state_dict(torch.load(model_path)["params"], strict=True)
    elif model_id == 19:
        # Aselsan Research Team
        from models.team19_imdeception import IMDeception
        name, data_range = f"{model_id:02}_IMDeception", 1.0
        model_path = os.path.join('model_zoo', 'team19_imdeception.pth')
        model = IMDeception(in_ch=3, scale=4, core=16, out_ch=3)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 20:
        # NJUST_ESR Team
        from models.team20_mobilesr import MobileSR
        name, data_range = f"{model_id:02}_MobileSR", 1.0
        model_path = os.path.join('model_zoo', 'team20_mobilesr.pth')
        model = MobileSR()
        model.load_state_dict(torch.load(model_path)["net"], strict=True)
    elif model_id == 21:
        # cceNBgdd Team
        pass
    elif model_id == 22:
        # Bilibili AI Team
        from models.team22_rep_rfdn import RFDN40
        name, data_range = f"{model_id:02}_RFDN40", 1.0
        model_path = os.path.join('model_zoo', 'team22_rep_rfdn.pth')
        model = RFDN40()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 23:
        # ncepu_explorers Team
        from models.team23_mdan.mdan import MDAN
        name, data_range = f"{model_id:02}_MDAN", 255.0
        model_path = os.path.join('model_zoo', 'team23_mdan.pt')
        model = MDAN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 24:
        # Strong Tiger Team
        from models.team24_mdgn import MDGN
        name, data_range = f"{model_id:02}_MDGN", 255.0
        model_path = os.path.join('model_zoo', 'team24_mdgn.pth')
        model = MDGN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 25:
        # TOVBU Team
        from models.team25_frfdn.FRFDN import FasterRFDN
        name, data_range = f"{model_id:02}_FasterRFDN", 1.0
        model_path = os.path.join('model_zoo', 'team25_frfdn.pth')
        model = FasterRFDN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 26:
        # xilinxSR Team
        from models.imdn_baseline import IMDN
        name, data_range = f"{model_id:02}_IMDN", 1.0
        model_path = os.path.join('model_zoo', 'team26_imdn_nb7.pth')
        model = IMDN(in_nc=3, out_nc=3, nc=64, nb=7, upscale=4, act_mode='L', upsample_mode='pixelshuffle')
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 27:
        # Just Try Team
        from models.team27_lwfanet import LWFANet
        name, data_range = f"{model_id:02}_LWFANet", 1.0
        model_path = os.path.join('model_zoo', 'team27_lwfanet.pth')
        model = LWFANet(num_in_ch=3, num_out_ch=3, num_feat=96, num_block=10)
        model.load_state_dict(torch.load(model_path)["params"], strict=True)
    elif model_id == 28:
        # neptune Team
        from models.team28_nasnetbn import NASNetBN
        name, data_range = f"{model_id:02}_NASNetBN", 1.0
        model_path = os.path.join('model_zoo', 'team28_nasnetbn.pth')
        model = NASNetBN(in_nc=3, out_nc=3, nf=32, nb=16, upscale=4,
                         arch_list=[3, 1, 2, 3, 3, 0, 1, 2, 0, 0, 0, 0, 2, 3, 3, 1])
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 29:
        # VAP Team
        from models.team29_clrfdn import RFDN_Conv3X3
        name, data_range = f"{model_id:02}_RFDN_Conv3X3", 255.0
        model_path = os.path.join('model_zoo', 'team29_clrfdn.pth')
        model = RFDN_Conv3X3(upscale=4)
        state_dict = torch.load(model_path)
        new_state_dict = dict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        model.load_state_dict(new_state_dict, strict=True)
    elif model_id == 30:
        # Giantpandacv Team
        from models.team30_scet import SCET
        name, data_range = f"{model_id:02}_SCET", 1.0
        model_path = os.path.join('model_zoo', 'team30_scet.pth')
        model = SCET(64, 128, 4)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 31:
        # Alpan Team
        from models.team31_sr_model import SR_model
        name, data_range = f"{model_id:02}_SR_model", 1.0
        model_path = os.path.join('model_zoo', 'team31_sr_model.pth')
        model = SR_model()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 32:
        # TeamInception
        pass
    elif model_id == 33:
        # Multicog Team
        from models.team33_m_rfdn.m_RFDN import m_RFDN
        name, data_range = f"{model_id:02}_m_RFDN", 1.0
        model_path = os.path.join('model_zoo', 'team33_m_rfdn.pth')
        model = m_RFDN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 34:
        # Dragon Team
        from models.team34_esan import make_model
        name, data_range = f"{model_id:02}_ESAN", 255.0
        model_path = os.path.join('model_zoo', 'team34_esan.pt')
        model = make_model(1) #.to(torch.device('cuda'))
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 35:
        # Set5baby Team
        from models.team35_rfdn.rfdn import RFDN
        name, data_range = f"{model_id:02}_RFDN", 255.0
        model_path = os.path.join('model_zoo', 'team35_rfdn.pt')
        model = RFDN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 36:
        # imglhl
        from models.team36_rfesr import RFESR
        name, data_range = f"{model_id:02}_RFESR", 255.0
        model_path = os.path.join('model_zoo', 'team36_rfesr.pt')
        model = RFESR(in_nc=3, nf=32, num_modules=4, out_nc=3, upscale=4)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    elif model_id == 37:
        # NWPU_SweetDreamLab
        from models.team37_bmdn import BMDN
        name, data_range = f"{model_id:02}_BMDN", 1.0
        model_path = os.path.join('model_zoo', 'team37_bmdn.pth')
        model = BMDN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 38:
        # SSL Team
        from models.team38_rfdnext.RFDN import RFDN
        name, data_range = f"{model_id:02}_RFDN", 1.0
        model_path = "model_zoo/team38_rfdnext.pth"
        model = RFDN(block_type="RFDB", act_type="lrelu")
        model.load_state_dict(torch.load(model_path)["model_state_dict"], strict=True)
    elif model_id == 39:
        # rainbow Team
        from models.team39_imdn_plus import IMDN_plus
        name, data_range = f"{model_id:02}_IMDN_plus", 1.0
        model_path = os.path.join('model_zoo', 'team39_imdn_plus.pth')
        model = IMDN_plus(in_nc=3, nf=36, nb=8, out_nc=3)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 40:
        # MegSR Team
        from models.team40_rfdn_pruned import RFDN as RFDNPrune
        name, data_range = f"{model_id:02}_RFDNPrune", 255.0
        model_path = os.path.join('model_zoo', 'team40_rfdn_pruned.pth')
        model = RFDNPrune(in_nc=3, nf=40, num_modules=4, out_nc=3, upscale=4)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 41:
        pass
    elif model_id == 42:
        # IMGWLH Team
        from models.team42_rlcsr import RLCSR
        name, data_range = f"{model_id:02}_RLCSR", 255.0
        model_path = os.path.join('model_zoo', 'team42_rlcsr.pt')
        model = RLCSR(in_nc=3, nf=32, num_modules=6, out_nc=3, upscale=4)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 43:
        # cipher Team
        from models.team43_resdn import ResDN
        name, data_range = f"{model_id:02}_ResDN", 1.0
        model_path = os.path.join('model_zoo', 'team43_resdn.pth')
        model = ResDN(upscale_factor=4, in_channels=3, n_feats=48, out_channels=3)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 44:
        # VMCL_Taobao Team
        from models.team44_msdn import MSDN
        name, data_range = f"{model_id:02}_MSDN", 1.0
        model_path = os.path.join('model_zoo', 'team44_msdn.pth')
        model = MSDN(in_nc=3, nf=56, dist_rate=0.5, num_modules=3, out_nc=3, upscale=4, act_type='silu')
        model.load_state_dict(torch.load(model_path), strict=True)
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    # print(model)
    model.eval()
    tile = 256 if model_id == 2 else None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model, name, data_range, tile


def select_dataset(data_dir, mode):
    if mode == "test":
        path = [
            (
                os.path.join(data_dir, f"DIV2K_test_LR/{i:04}.png"),
                os.path.join(data_dir, f"DIV2K_test_HR/{i:04}.png")
            ) for i in range(901, 1001)
        ]
        # [f"DIV2K_test_LR/{i:04}.png" for i in range(901, 1001)]
    else:
        path = [
            (
                os.path.join(data_dir, f"DIV2K_valid_LR/{i:04}x4.png"),
                os.path.join(data_dir, f"DIV2K_valid_HR/{i:04}.png")
            ) for i in range(801, 901)
        ]

    return path


def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


def run(model, model_name, data_range, tile, logger, device, args, mode="test"):

    sf = 4
    border = sf
    results = dict()
    results[f"{mode}_runtime"] = []
    results[f"{mode}_psnr"] = []
    if args.ssim:
        results[f"{mode}_ssim"] = []
    # results[f"{mode}_psnr_y"] = []
    # results[f"{mode}_ssim_y"] = []

    # --------------------------------
    # dataset path
    # --------------------------------
    data_path = select_dataset(args.data_dir, mode)
    save_path = os.path.join(args.save_dir, model_name, "test" if mode == "test" else "valid")
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i, (img_lr, img_hr) in enumerate(data_path):

        # --------------------------------
        # (1) img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_hr))
        img_lr = util.imread_uint(img_lr, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)

        # --------------------------------
        # (2) img_sr
        # --------------------------------
        start.record()
        img_sr = forward(img_lr, model, tile)
        end.record()
        torch.cuda.synchronize()
        results[f"{mode}_runtime"].append(start.elapsed_time(end))  # milliseconds
        img_sr = util.tensor2uint(img_sr, data_range)

        # --------------------------------
        # (3) img_hr
        # --------------------------------
        img_hr = util.imread_uint(img_hr, n_channels=3)
        img_hr = img_hr.squeeze()
        img_hr = util.modcrop(img_hr, sf)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        psnr = util.calculate_psnr(img_sr, img_hr, border=border)
        results[f"{mode}_psnr"].append(psnr)

        if args.ssim:
            ssim = util.calculate_ssim(img_sr, img_hr, border=border)
            results[f"{mode}_ssim"].append(ssim)
            logger.info("{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.".format(img_name + ext, psnr, ssim))
        else:
            logger.info("{:s} - PSNR: {:.2f} dB".format(img_name + ext, psnr))

        # if np.ndim(img_hr) == 3:  # RGB image
        #     img_sr_y = util.rgb2ycbcr(img_sr, only_y=True)
        #     img_hr_y = util.rgb2ycbcr(img_hr, only_y=True)
        #     psnr_y = util.calculate_psnr(img_sr_y, img_hr_y, border=border)
        #     ssim_y = util.calculate_ssim(img_sr_y, img_hr_y, border=border)
        #     results[f"{mode}_psnr_y"].append(psnr_y)
        #     results[f"{mode}_ssim_y"].append(ssim_y)

        util.imsave(img_sr, os.path.join(save_path, img_name[:4]+ext))

    results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2  # !
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"])  # !
    results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])
    if args.ssim:
        results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])
    # results[f"{mode}_ave_psnr_y"] = sum(results[f"{mode}_psnr_y"]) / len(results[f"{mode}_psnr_y"])
    # results[f"{mode}_ave_ssim_y"] = sum(results[f"{mode}_ssim_y"]) / len(results[f"{mode}_ssim_y"])
    logger.info("{:>16s} : {:<.3f} [M]".format("Max Memery", results[f"{mode}_memory"]))  # Memery
    logger.info("------> Average runtime of ({}) is : {:.6f} seconds".format("test" if mode == "test" else "valid", results[f"{mode}_ave_runtime"]))

    return results


def main(args):

    utils_logger.logger_info("NTIRE2022-EfficientSR", log_path="NTIRE2022-EfficientSR.log")
    logger = logging.getLogger("NTIRE2022-EfficientSR")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model, model_name, data_range, tile = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if True:
        # --------------------------------
        # restore image
        # --------------------------------

        # inference on the validation set
        valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="valid")
        # record PSNR, runtime
        results[model_name] = valid_results

        if args.include_test:
            # inference on the test set
            test_results = run(model, model_name, data_range, tile, logger, device, args, mode="test")
            results[model_name].update(test_results)

        input_dim = (3, 256, 256)  # set the input dimension
        activations, num_conv = get_model_activation(model, input_dim)  # !
        activations = activations/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
        logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

        flops = get_model_flops(model, input_dim, False)  # !
        flops = flops/10**9
        logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))  # !
        num_parameters = num_parameters/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
        results[model_name].update({"activations": activations, "num_conv": num_conv, "flops": flops, "num_parameters": num_parameters})

        with open(json_dir, "w") as f:
            json.dump(results, f)
    if args.include_test:
        fmt = "{:20s}\t{:10s}\t{:10s}\t{:14s}\t{:14s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Test PSNR", "Val Time [ms]", "Test Time [ms]", "Ave Time [ms]",
                       "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    else:
        fmt = "{:20s}\t{:10s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Val Time [ms]", "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    for k, v in results.items():
        val_psnr = f"{v['valid_ave_psnr']:2.2f}"
        val_time = f"{v['valid_ave_runtime']:3.2f}"
        num_param = f"{v['num_parameters']:2.3f}"
        flops = f"{v['flops']:2.2f}"
        acts = f"{v['activations']:2.2f}"
        mem = f"{v['valid_memory']:2.2f}"
        conv = f"{v['num_conv']:4d}"
        if args.include_test:
            # from IPython import embed; embed()
            test_psnr = f"{v['test_ave_psnr']:2.2f}"
            test_time = f"{v['test_ave_runtime']:3.2f}"
            ave_time = f"{(v['valid_ave_runtime'] + v['test_ave_runtime']) / 2:3.2f}"
            s += fmt.format(k, val_psnr, test_psnr, val_time, test_time, ave_time, num_param, flops, acts, mem, conv)
        else:
            s += fmt.format(k, val_psnr, val_time, num_param, flops, acts, mem, conv)
    with open(os.path.join(os.getcwd(), 'results.txt'), "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2022-EfficientSR")
    parser.add_argument("--data_dir", default="/cluster/work/cvl/yawli/data/NTIRE2022_Challenge", type=str)
    parser.add_argument("--save_dir", default="/cluster/work/cvl/yawli/data/NTIRE2022_Challenge/results", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--include_test", action="store_true", help="Inference on the DIV2K test set")
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")

    args = parser.parse_args()
    pprint(args)

    main(args)
