import argparse
import os
import glob

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import albumentations as albu
import segmentation_models_pytorch as smp

from table.utils import ensure_color
from lab.loader import (
    CompSegDataset,
    get_preprocessing,
    get_training_augmentation,
    get_validation_augmentation
)


def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = (
    "sigmoid"  # could be None for logits or 'softmax2d' for multicalss segmentation
)
DEVICE = "cuda"
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


class Trainer:
    def __init__(self, args):
        self.data_dir="/home/luan/research/compseg/data1/train"
        self.fold_idx=1


    def get_train_loader(self):
        train_dataset = CompSegDataset(
            root_dir=self.data_dir,
            fold_idx=self.fold_idx,
            stage="train",
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
        )

        print(f"Len train set: {len(train_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=8
        )
        return train_loader

    def get_val_loader(self):
        val_dataset = CompSegDataset(
            root_dir=self.data_dir,
            fold_idx=self.fold_idx,
            stage="val",
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
        )

        print(f"Len val set: {len(val_dataset)}")

        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
        return val_loader

    def get_model(self):
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=1,
            activation=ACTIVATION,
        )
        return model
    
    def run(self):
        loss = smp.utils.losses.DiceLoss()
        metrics = [smp.utils.metrics.IoU(threshold=0.5),]

        model = self.get_model()

        optimizer = optim.Adam([dict(
            params=model.parameters(),
            lr=0.0001
        ),])
        
        train_loader = self.get_train_loader()
        valid_loader = self.get_val_loader()

        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True,
        )

        max_score = 0
        for i in range(0, 10):
            print("\nEpoch: {}".format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs["iou_score"]:
                max_score = valid_logs["iou_score"]
                torch.save(model, "./best_model.pth")
                print("Model saved!")

            if i == 5:
                optimizer.param_groups[0]["lr"] = 1e-5
                print("Decrease decoder learning rate to 1e-5!")


def main():
    args = get_args()
    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
