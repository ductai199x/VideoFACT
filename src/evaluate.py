import sys
import os

sys.path.insert(0, os.getcwd())

import argparse
import yaml

from typing import *

from lightning.pytorch import Trainer

from data.load_single_dataset import load_single_dataset
from data.load_single_video import load_single_video
from data.process_single_video import process_single_video
from model.prepare_model import prepare_model

parser = argparse.ArgumentParser()

available_datasets = [
    "vcms",
    "vpvm",
    "vpim",
    "videosham",
    "inpainting",
    "deepfake",
    "icms",
    "ipvm",
    "ipim",
]


def get_trainer(args):
    return Trainer(
        accelerator="cpu" if args.cpu else "gpu",
        devices=1,
        logger=None,
        profiler=None,
        callbacks=None,
    )


def eval_dataset(args):
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        ablation_codename = configs["ablation_codename"]

    data = load_single_dataset(
        args.dataset,
        args.dataset_type,
        args.shuffle,
        args.num_samples,
        args.contains,
        args.batch_size,
        args.num_workers,
    )
    model = prepare_model(
        ablation_codename,
        args.prev_ckpt,
        configs,
    )
    trainer = get_trainer(args)
    trainer.test(model, data)


def parse_eval_dataset():
    global parser, subparsers
    p = subparsers.add_parser("dataset", help="Evaluate model on a single dataset")
    p.add_argument(
        "--dataset",
        choices=available_datasets,
        type=str,
        required=True,
    )
    p.add_argument(
        "--dataset_type",
        choices=["train", "val", "test"],
        type=str,
        default="test",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=10,
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=10,
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=-1,
    )
    p.add_argument(
        "--contains",
        type=str,
        default="",
    )
    p.set_defaults(func=eval_dataset)


def eval_video(args):
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        ablation_codename = configs["ablation_codename"]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataloader = load_single_video(
        args.video_path,
        args.shuffle,
        args.max_num_samples,
        args.sample_every,
        args.batch_size,
        args.num_workers,
    )
    model = prepare_model(
        ablation_codename,
        args.prev_ckpt,
        configs,
    )
    classification_stats = process_single_video(model, dataloader, args.output_dir, args.cpu)
    print(classification_stats)


def parse_eval_video():
    global parser, subparsers
    p = subparsers.add_parser("video", help="Evaluate model on a single dataset")
    p.add_argument(
        "--video_path",
        type=str,
        required=True,
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=10,
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=10,
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
    )
    p.add_argument(
        "--max_num_samples",
        type=int,
        default=-1,
    )
    p.add_argument(
        "--sample_every",
        type=int,
        default=1,
    )
    p.set_defaults(func=eval_video)


def parse_args():
    global parser, subparsers, ARGS
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="the path to a config file",
        default=None,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--prev-ckpt",
        type=str,
        help="the path to a previous checkpoint",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="use cpu for inference",
    )

    subparsers = parser.add_subparsers()
    parse_eval_dataset()
    parse_eval_video()

    ARGS = parser.parse_args()
    ARGS.func(ARGS)


def main():
    parse_args()


if __name__ == "__main__":
    main()
