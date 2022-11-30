import sys

sys.path.append("..")

from typing import List

from torch.utils.data import ConcatDataset, DataLoader

from utils import get_all_files, rand_split

from .common import CommonImageDataset
from .dataset_paths import DATASET_ROOT_PATH


def get_dataset(dataset_name: str, dataset_samples: List[str], **args):
    return CommonImageDataset(dataset_samples, **args)
    # dataset_name = dataset_name.lower()
    # if dataset_name in ("vcms", "vpvm", "vpim", "icms", "ipvm", "ipim"):

    # elif dataset_name == "videosham":
    #     return VideoShamAdobeDataset(dataset_samples, **args)
    # elif dataset_name == "inpainting":
    #     return E2fgviDavisDataset(dataset_samples, **args)
    # else:
    #     raise NotImplementedError


def load_training_data(dataset_name, batch_size=3, num_workers=6):
    function_name = f"load_data_{dataset_name}"
    if function_name in globals():
        dataset_function = globals()[function_name]
        return dataset_function(batch_size, num_workers)
    else:
        raise ValueError(f"{dataset_name} does not exists. Please recheck.")


def load_data_vcms(batch_size, num_workers):
    train_vcms_samples = get_all_files(f"{DATASET_ROOT_PATH['vcms']}/train", suffix=".png")
    val_vcms_samples = get_all_files(f"{DATASET_ROOT_PATH['vcms']}/val", suffix=".png")

    train_vcms_ds = get_dataset("vcms", train_vcms_samples)
    val_vcms_ds = get_dataset("vcms", val_vcms_samples)

    train_dl = DataLoader(train_vcms_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dl = DataLoader(val_vcms_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_dl, val_dl


def load_data_vpvm(batch_size, num_workers):
    train_vpvm_samples = get_all_files(f"{DATASET_ROOT_PATH['vpvm']}/train", suffix=".png")
    val_vpvm_samples = get_all_files(f"{DATASET_ROOT_PATH['vpvm']}/val", suffix=".png")

    train_vpvm_ds = get_dataset("vpvm", train_vpvm_samples)
    val_vpvm_ds = get_dataset("vpvm", val_vpvm_samples)

    train_dl = DataLoader(train_vpvm_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dl = DataLoader(val_vpvm_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_dl, val_dl


def load_data_vpim(batch_size, num_workers):
    train_vpim_samples = get_all_files(f"{DATASET_ROOT_PATH['vpim']}/train", suffix=".png")
    val_vpim_samples = get_all_files(f"{DATASET_ROOT_PATH['vpim']}/val", suffix=".png")

    train_vpim_ds = get_dataset("vpim", train_vpim_samples)
    val_vpim_ds = get_dataset("vpim", val_vpim_samples)

    train_dl = DataLoader(train_vpim_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dl = DataLoader(val_vpim_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_dl, val_dl


def load_data_mixed_all_videos(batch_size, num_workers):
    train_vcms_samples = get_all_files(f"{DATASET_ROOT_PATH['vcms']}/train", suffix=".png")
    train_vpvm_samples = get_all_files(f"{DATASET_ROOT_PATH['vpvm']}/train", suffix=".png")
    train_vpim_samples = get_all_files(f"{DATASET_ROOT_PATH['vpim']}/train", suffix=".png")
    val_vcms_samples = get_all_files(f"{DATASET_ROOT_PATH['vcms']}/val", suffix=".png")
    val_vpvm_samples = get_all_files(f"{DATASET_ROOT_PATH['vpvm']}/val", suffix=".png")
    val_vpim_samples = get_all_files(f"{DATASET_ROOT_PATH['vpim']}/val", suffix=".png")

    train_vcms_ds = get_dataset("vcms", train_vcms_samples)
    train_vpvm_ds = get_dataset("vpvm", train_vpvm_samples)
    train_vpim_ds = get_dataset("vpim", train_vpim_samples)
    val_vcms_ds = get_dataset("vcms", val_vcms_samples)
    val_vpvm_ds = get_dataset("vpvm", val_vpvm_samples)
    val_vpim_ds = get_dataset("vpim", val_vpim_samples)

    train_vcms_ds, _ = rand_split(train_vcms_ds, 0.1)
    train_vpvm_ds, _ = rand_split(train_vpvm_ds, 0.1)
    train_vpim_ds, _ = rand_split(train_vpim_ds, 0.8)
    val_vcms_ds, _ = rand_split(val_vcms_ds, 0.1)
    val_vpvm_ds, _ = rand_split(val_vpvm_ds, 0.1)
    val_vpim_ds, _ = rand_split(val_vpim_ds, 0.8)

    train_ds = ConcatDataset(
        [
            train_vcms_ds,
            train_vpvm_ds,
            train_vpim_ds,
        ]
    )
    val_ds = ConcatDataset(
        [
            val_vcms_ds,
            val_vpvm_ds,
            val_vpim_ds,
        ]
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_dl, val_dl


def load_data_mixed_all_videos_and_images(batch_size, num_workers):
    train_vcms_samples = get_all_files(f"{DATASET_ROOT_PATH['vcms']}/train", suffix=".png")
    train_vpvm_samples = get_all_files(f"{DATASET_ROOT_PATH['vpvm']}/train", suffix=".png")
    train_vpim_samples = get_all_files(f"{DATASET_ROOT_PATH['vpim']}/train", suffix=".png")
    train_icms_samples = get_all_files(f"{DATASET_ROOT_PATH['icms']}/train", suffix=".png")
    train_ipvm_samples = get_all_files(f"{DATASET_ROOT_PATH['ipvm']}/train", suffix=".png")
    train_ipim_samples = get_all_files(f"{DATASET_ROOT_PATH['ipim']}/train", suffix=".png")
    val_vcms_samples = get_all_files(f"{DATASET_ROOT_PATH['vcms']}/val", suffix=".png")
    val_vpvm_samples = get_all_files(f"{DATASET_ROOT_PATH['vpvm']}/val", suffix=".png")
    val_vpim_samples = get_all_files(f"{DATASET_ROOT_PATH['vpim']}/val", suffix=".png")
    val_icms_samples = get_all_files(f"{DATASET_ROOT_PATH['icms']}/val", suffix=".png")
    val_ipvm_samples = get_all_files(f"{DATASET_ROOT_PATH['ipvm']}/val", suffix=".png")
    val_ipim_samples = get_all_files(f"{DATASET_ROOT_PATH['ipim']}/val", suffix=".png")

    train_vcms_ds = get_dataset("vcms", train_vcms_samples)
    train_vpvm_ds = get_dataset("vpvm", train_vpvm_samples)
    train_vpim_ds = get_dataset("vpim", train_vpim_samples)
    train_icms_ds = get_dataset("icms", train_icms_samples)
    train_ipvm_ds = get_dataset("ipvm", train_ipvm_samples)
    train_ipim_ds = get_dataset("ipim", train_ipim_samples)
    val_vcms_ds = get_dataset("vcms", val_vcms_samples)
    val_vpvm_ds = get_dataset("vpvm", val_vpvm_samples)
    val_vpim_ds = get_dataset("vpim", val_vpim_samples)
    val_icms_ds = get_dataset("icms", val_icms_samples)
    val_ipvm_ds = get_dataset("ipvm", val_ipvm_samples)
    val_ipim_ds = get_dataset("ipim", val_ipim_samples)

    train_vcms_ds, _ = rand_split(train_vcms_ds, 0.1)
    train_vpvm_ds, _ = rand_split(train_vpvm_ds, 0.1)
    train_vpim_ds, _ = rand_split(train_vpim_ds, 0.3)
    val_vcms_ds, _ = rand_split(val_vcms_ds, 0.1)
    val_vpvm_ds, _ = rand_split(val_vpvm_ds, 0.1)
    val_vpim_ds, _ = rand_split(val_vpim_ds, 0.3)

    train_icms_ds, _ = rand_split(train_icms_ds, 0.1)
    train_ipvm_ds, _ = rand_split(train_ipvm_ds, 0.1)
    train_ipim_ds, _ = rand_split(train_ipim_ds, 0.7)
    val_icms_ds, _ = rand_split(val_icms_ds, 0.1)
    val_ipvm_ds, _ = rand_split(val_ipvm_ds, 0.1)
    val_ipim_ds, _ = rand_split(val_ipim_ds, 0.7)

    train_ds = ConcatDataset(
        [
            train_vcms_ds,
            train_vpvm_ds,
            train_vpim_ds,
            train_icms_ds,
            train_ipvm_ds,
            train_ipim_ds,
        ]
    )
    val_ds = ConcatDataset(
        [
            val_vcms_ds,
            val_vpvm_ds,
            val_vpim_ds,
            val_icms_ds,
            val_ipvm_ds,
            val_ipim_ds,
        ]
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_dl, val_dl
