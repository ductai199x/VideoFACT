import sys

sys.path.append("..")

import random
from typing import Literal

from torch.utils.data import DataLoader

from utils import get_all_files

from .common import CommonImageDataset
from .dataset_paths import DATASET_ROOT_PATH
from .e2fgvi_inpainting import E2fgviDavisDataset
from .videosham import VideoShamAdobeDataset


def load_single_dataset(
    dataset_name,
    dataset_type: Literal["train", "val", "test"] = "test",
    shuffle=False,
    num_samples=-1,
    filepath_contains="",
    batch_size=3,
    num_workers=6,
    **args,
):
    dataset_name = dataset_name.lower()
    if dataset_name in ("vcms", "vpvm", "vpim", "icms", "ipvm", "ipim"):
        dataset_samples = get_all_files(
            f"{DATASET_ROOT_PATH[dataset_name]}/{dataset_type}", suffix=".png", contains=filepath_contains
        )
        if shuffle:
            random.shuffle(dataset_samples)
        if num_samples > 0:
            dataset_samples = dataset_samples[:num_samples]
        dataset = CommonImageDataset(dataset_samples, **args)
    elif dataset_name == "videosham":
        dataset_samples = get_all_files(
            f"{DATASET_ROOT_PATH['videosham']}", suffix=".png", contains=filepath_contains
        )
        if shuffle:
            random.shuffle(dataset_samples)
        if num_samples > 0:
            dataset_samples = dataset_samples[:num_samples]
        dataset = VideoShamAdobeDataset(dataset_samples, **args)
    elif dataset_name == "inpainting":
        assert (
            "res_divider" in args
        ), "You must specify the resolution divider for the inpainting dataset. res_divider=[1, 2, 3, 4]"
        resolution = (1080 // args["res_divider"], 1920 // args["res_divider"])
        dataset_samples = get_all_files(
            f"{DATASET_ROOT_PATH['inpainting']}/ds_{resolution[1]}x{resolution[0]}",
            suffix=(".png", ".jpg"),
            contains=filepath_contains,
        )
        if shuffle:
            random.shuffle(dataset_samples)
        if num_samples > 0:
            dataset_samples = dataset_samples[:num_samples]
        dataset = E2fgviDavisDataset(dataset_samples, **args)
    else:
        raise NotImplementedError

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
