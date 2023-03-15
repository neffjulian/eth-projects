import os
from argparse import ArgumentParser
from pathlib import Path

import time
import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from src.dataset import RoadSegDataModule
from src.UNet import SMPModel
from src.utils import object_from_dict


def save_predictions(batches):
    pred_path = "data/processed/test/predictions"
    os.makedirs(pred_path, exist_ok=True)

    img_names = []
    preds = []
    for names, images in batches:
        img_names += list(names)
        preds += images

    preds = torch.cat(preds, dim=0)

    for name, img in zip(img_names, preds):
        img = img*255
        img = img.squeeze().numpy()
        img = A.CenterCrop(400, 400, always_apply=True)(image=img)["image"]
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img_path = os.path.join(pred_path, name)
        img.save(img_path)


def run_experiment(hparams):
    seed_everything(hparams["random_seed"])

    model = SMPModel(hparams)
    if "resume_from_checkpoint" in hparams:
        model = model.load_from_checkpoint(hparams["resume_from_checkpoint"])

    output_stride = model.model.encoder.output_stride

    experiment_name = "_".join(
        [time.strftime("%Y%m%d_%H%M%S"), hparams["experiment_name"]])

    wandb_logger = WandbLogger(
        name=experiment_name, project="cil2022")
    logger = CSVLogger(name=experiment_name, save_dir="logs")
    loggers = [wandb_logger, logger]

    #log_predictions_callback = LogPredictionsCallback(wandb_logger)
    checkpoint_callback = ModelCheckpoint(monitor="val_f1", mode="max")
    callbacks = [checkpoint_callback]

    trainer = object_from_dict(hparams["trainer"],
                               default_root_dir="logs",
                               logger=loggers,
                               callbacks=callbacks,
                               )

    datamodule = RoadSegDataModule(
        data_dir="data", random_seed=hparams["random_seed"], output_stride=output_stride, **hparams["dataset"])

    if hparams["train"]:
        trainer.fit(model, datamodule=datamodule)

    if hparams["predict"]:
        pred_batches = trainer.predict(model, datamodule=datamodule)
        save_predictions(pred_batches)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=Path,
                        help="The config file", required=True)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    run_experiment(hparams)
