import cv2
import torch  # noqa

from .back_prop import GaussianBlurPath, attribution_objective, Path_gradient, saliency_map_PG
from .utils import cv2_to_pil, pil_to_cv2, gini, vis_saliency, vis_saliency_kde, \
    click_select_position, grad_abs_norm, grad_norm, prepare_images, make_pil_grid, blend_input, attr_grad, \
    load_as_tensor, Tensor2PIL, PIL2Tensor


def get_model_interpretation(model, img_path, w=110, h=150, window_size=16,
                             sigma=1.2, fold=50, ll=9, alpha=0.5, use_cuda: bool = False):
    r"""From paper "Interpreting Super-Resolution Networks with Local Attribution Maps".
        Modified from "https://github.com/X-Lowlevel-Vision/LAM_Demo".

    Args:
        model:
        img_path:
        w: The x coordinate of your select patch, 125 as an example
        h: The y coordinate of your select patch, 160 as an example
        window_size: Window size of D
        sigma:
        fold:
        ll:
        alpha:
        use_cuda:

    """

    img_lr, img_hr = prepare_images(img_path)
    tensor_lr = PIL2Tensor(img_lr)[:3]

    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)  # noqa
    position_pil = cv2_to_pil(draw_img)

    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, ll)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective,
                                                                              gaus_blur_path_func, cuda=use_cuda)
    grad_numpy, result = saliency_map_PG(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)
    saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
    blend_abs_and_input = cv2_to_pil(
        pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    blend_kde_and_input = cv2_to_pil(
        pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    pil = make_pil_grid(
        [
            position_pil,
            saliency_image_abs,
            blend_abs_and_input,
            blend_kde_and_input,
            # Tensor2PIL(torch.clamp(result, min=0., max=1.))
        ]
    )

    gini_index = gini(abs_normed_grad_numpy)
    diffusion_index = (1 - gini_index) * 100

    return pil, diffusion_index
