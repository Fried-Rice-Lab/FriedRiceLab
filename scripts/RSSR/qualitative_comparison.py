import os

import imageio.v2 as imageio
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def sub_plot(
    canvas,
    compare_pth,
    method_name,
    dataset_name,
    scale,
    image_name,
    image_range,
    title=None,
    whole_img=False,
    small_img=False,
    pin_img=False,
    cut_range=(0, 0, 0, 0),
    color="black",
):
    if method_name == "HR":
        image_path = os.path.join(
            compare_pth,
            method_name,
            "visualization/inference/lsr",
            "UCM_{}".format(dataset_name),
            "x{}".format(scale),
            "{}{}.png".format(dataset_name, image_name),
        )
    else:
        image_path = os.path.join(
            compare_pth,
            method_name,
            "visualization/inference/lsr",
            "UCM_{}".format(dataset_name),
            "x{}".format(scale),
            "{}{}.png".format(dataset_name, image_name),
        )
    image = imageio.imread(image_path)

    # image
    if not whole_img:
        if not small_img:
            image_patch = image[
                image_range[0] * scale : image_range[0] * scale
                + image_range[2] * scale,
                image_range[1] * scale : image_range[1] * scale
                + image_range[3] * scale,
                :,
            ]
        else:
            image_patch = image[
                image_range[0] : image_range[0] + image_range[2],
                image_range[1] : image_range[1] + image_range[3],
                :,
            ]
    else:
        image_patch = image
        if cut_range != (0, 0, 0, 0):
            image_patch = image_patch[
                cut_range[0] : cut_range[0] + cut_range[2],
                cut_range[1] : cut_range[1] + cut_range[3],
                :,
            ]
        if pin_img:  # Only pin if small_img is false
            rect = patches.Rectangle(
                (
                    (image_range[1] * scale) - cut_range[1],
                    (image_range[0] * scale) - cut_range[0],
                ),
                image_range[3] * scale,
                image_range[2] * scale,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            canvas.add_patch(rect)

    # title
    if title is None:
        pass
    else:
        canvas.set_title(title, color=color)

    # patch
    # if whole_img:
    #     if image_name == 'img_062.png':
    #         axins = inset_axes(canvas, width="45%", height="41.7%",
    #                            bbox_to_anchor=(-0.5, 0, 1, 1),
    #                            bbox_transform=canvas.transAxes)
    #     else:
    #         axins = inset_axes(canvas, width="45%", height="41.7%",
    #                            bbox_to_anchor=(0, 0, 1, 1),
    #                            bbox_transform=canvas.transAxes)
    #     image_patch_ = image[image_range[0] * scale:image_range[0] * scale + image_range[2] * scale,
    #                    image_range[1] * scale:image_range[1] * scale + image_range[3] * scale, :]
    #     rect_ = patches.Rectangle((0, 0),
    #                               86,
    #                               132,
    #                               linewidth=1, edgecolor='r', facecolor='none')
    #     axins.add_patch(rect_)
    #     axins.imshow(image_patch_)
    #     axins.axis('off')

    canvas.imshow(image_patch)
    canvas.axis("off")


def draw_bi_compare(
    compare_pth,
    method_name,
    dataset_name: list,
    scale: int,
    image_name: list,
    image_range,
    cut_range,
    dpi=80,
    figsize=(8, 3),
):
    plt.figure(dpi=dpi, figsize=figsize)
    gs = gridspec.GridSpec(len(image_name) * 2, 7)

    for index, method in enumerate(method_name):
        method_name[index] = "{}".format(method)

    for image_index in range(len(image_name)):
        ax_rw = plt.subplot(gs[image_index * 2 : image_index * 2 + 2, 0:2])
        sub_plot(
            ax_rw,
            compare_pth,
            "CTHN",
            dataset_name[image_index],
            scale,
            image_name[image_index],
            image_range[image_index],
            whole_img=True,
            pin_img=True,
            cut_range=cut_range[image_index],
            title="{}: {}".format(
                dataset_name[image_index], "{}.png".format(image_name[image_index])
            ),
        )
        for method_index in range(len(method_name)):
            if method_index < 4:
                ax_rw = plt.subplot(gs[image_index * 2, method_index + 2])
                sub_plot(
                    ax_rw,
                    compare_pth,
                    method_name[method_index],
                    dataset_name[image_index],
                    scale,
                    image_name[image_index],
                    image_range[image_index],
                    title=method_name[method_index],
                )
            else:
                ax_rw = plt.subplot(gs[image_index * 2 + 1, method_index - 2])
                sub_plot(
                    ax_rw,
                    compare_pth,
                    method_name[method_index],
                    dataset_name[image_index],
                    scale,
                    image_name[image_index],
                    image_range[image_index],
                    title=method_name[method_index],
                )

    plt.show()


if __name__ == "__main__":
    compare_path_ = "~/FriedRiceLab/results"

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    BI_list = [
        "EDSR",
        "LAPAR",
        "FDIWN",
        "BSRN",  # CNN-based methods
        "SwinIR",
        "ESRT",  # Transformer-based methods
        "CTHN",  # Ours
        "HR",
    ]

    draw_bi_compare(
        compare_pth=compare_path_,
        method_name=BI_list,
        scale=4,
        dataset_name=[
            "agricultural",
            "airplane",
            "intersection",
            "overpass",
        ],
        image_name=[
            "21",
            "84",
            "47",
            "46",
        ],
        image_range=[
            [100, 180, 20, 30],
            [30, 160, 20, 30],
            [40, 110, 20, 30],
            [100, 180, 20, 30],
        ],
        cut_range=[
            (50, 220, 495, 640),
            (50, 220, 495, 640),
            (50, 220, 495, 640),
            (50, 220, 495, 640),
        ],
        dpi=100,
        figsize=(9, 9),
    )

