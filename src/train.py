import sys
import os

sys.path.insert(0, os.getcwd())

import argparse
import json
import yaml

from ctypes import ArgumentError
from typing import *

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from data.load_training_data import load_training_data
from model.prepare_model import prepare_model

parser = argparse.ArgumentParser()
MODEL_CONFIG = dict()


def prepare_datasets():
    train_dl, val_dl = load_training_data(ARGS.custom_dataset_name)
    return train_dl, val_dl


def train():
    # define how the model is loaded in the prepare_model.py file
    model = prepare_model(ARGS.ablation_codename, ARGS.prev_ckpt, MODEL_CONFIG)
    train_dl, val_dl = prepare_datasets()

    logger = (
        None
        if ARGS.fast_dev_run
        else TensorBoardLogger(
            save_dir=os.getcwd(),
            version=f"version_{ARGS.version}",
            name=ARGS.log_dir,
            log_graph=True,
        )
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_ckpt = ModelCheckpoint(
        dirpath=f"{ARGS.log_dir}/version_{ARGS.version}/checkpoints",
        monitor="val_class_acc_epoch",
        filename=f"{ARGS.pre}-{{epoch:02d}}-{{val_class_acc_epoch:.4f}}",
        verbose=True,
        save_last=True,
        save_top_k=1,
        mode="max",
    )
    callbacks = (
        [ModelSummary(-1), TQDMProgressBar(refresh_rate=1), model_ckpt] + []
        if ARGS.fast_dev_run
        else [lr_monitor]
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=ARGS.gpus,
        max_epochs=ARGS.max_epoch,
        resume_from_checkpoint=ARGS.prev_ckpt if eval(ARGS.resume) else None,
        logger=logger,
        profiler=None,
        callbacks=callbacks,
        fast_dev_run=ARGS.fast_dev_run,
    )

    model(model.example_input_array)  # to get the model's graph to display in tblogger
    trainer.fit(model, train_dl, val_dl)


def parse_args():
    global MODEL_CONFIG, TRAIN_TFR, VAL_TFR
    # Verify all the run flags...
    if ARGS.config and not os.path.exists(ARGS.config):
        raise FileNotFoundError(f"config file does not exist: {ARGS.config}")
    if ARGS.prev_ckpt and not os.path.exists(ARGS.prev_ckpt):
        raise FileNotFoundError(f"previous checkpoint file does not exist: {ARGS.prev_ckpt}")
    if ARGS.log_dir and not os.path.isdir(ARGS.log_dir):
        print(f"Log dir does not exist: {ARGS.log_dir}. Trying to create it..")
        os.makedirs(ARGS.log_dir)
    if not eval(ARGS.force_version) and os.path.isdir(f"{ARGS.log_dir}/version_{ARGS.version}"):
        prompt = input(
            f"Version {ARGS.version} already exist in {ARGS.log_dir}. Do you want to continue? (y/n)"
        )
        if prompt.lower().strip()[0] == "y":
            pass
        else:
            sys.exit()
    if eval(ARGS.resume) and not ARGS.prev_ckpt:
        raise ArgumentError("resume is true but there's no checkpoint specified")

    with open(ARGS.config, "r") as f:
        ext_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k in ext_conf:
            MODEL_CONFIG[k] = ext_conf[k]
        ARGS.__setattr__("custom_dataset_name", MODEL_CONFIG["custom_dataset_name"])
        ARGS.__setattr__("ablation_codename", MODEL_CONFIG["ablation_codename"])

    print(json.dumps(MODEL_CONFIG, indent=4))


def main():
    global ARGS
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        help="the number of example per batch",
        default=4,
    )
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
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        help="resume the training progress? False will reset the optimizer state (True/False)",
        default="False",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="the version of this model (same as the one saved in log dir",
        default=0,
    )
    parser.add_argument(
        "--force-version",
        type=str,
        help="ignore version conflict",
        default="False",
    )
    parser.add_argument(
        "-l",
        "--log-dir",
        type=str,
        help="the path to the log directory",
        default="src/lightning_logs",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        help="the number of gpus used for training",
        default=1,
    )
    parser.add_argument(
        "-e",
        "--max-epoch",
        type=int,
        help="max number of training epoch",
        default=20,
    )
    parser.add_argument(
        "-d",
        "--fast-dev-run",
        action="store_true",
        help="fast dev run? (True/Fase)",
    )
    parser.add_argument(
        "--pre",
        type=str,
        help="checkpoint's prefix",
        default="",
    )
    ARGS = parser.parse_args()
    parse_args()
    train()


if __name__ == "__main__":
    main()
