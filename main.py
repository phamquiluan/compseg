import argparse
import os
import glob
import json
import datetime

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import albumentations as albu
import segmentation_models_pytorch as smp

from table.utils import ensure_color
from lab.loader import (
    CompSegDataset,
    HairDataset,
    get_preprocessing,
    get_training_augmentation,
)

cv2.setNumThreads(8)

ENCODER_WEIGHTS = "imagenet"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold-idx', type=int, default=1)

    parser.add_argument('--arch', default="unet")
    parser.add_argument('--encoder', default="resnet34")
    parser.add_argument('--image-size', type=int, default=512)

    parser.add_argument('--num-epochs', default=100)
    parser.add_argument('--lr', default=0.0001)
    parser.add_argument('--step-size', default=80)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)

    parser.add_argument('--tta', action="store_true")
    # parser.add_argument('--data-dir', default="/home/luan/research/compseg/data/sid") 
    parser.add_argument('--data-dir', default="/home/luan/research/compseg/data/kane") 

    args = parser.parse_args()
    return args

ACTIVATION = ("sigmoid")
DEVICE = "cuda"
st = datetime.datetime.now()
st_str = st.strftime("%m%d_%H%M")

args = get_args()


class Trainer:
    # wth has been done here
    # input space (BGR), input range, mean, std, howabout image size?
    # could we turn separately??
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, ENCODER_WEIGHTS)

    # resnet 34, imagenet
    preprocessing = get_preprocessing(preprocessing_fn)

    def __init__(self, args):
        self.args = args
        self.data_dir = args.data_dir
        self.fold_idx = args.fold_idx
        self.crop_size = int(0.625 * args.image_size)

        log_dir = f"kane_log/{args.arch}_{args.encoder}_fold_{args.fold_idx}_{st_str}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)


    def get_train_loader(self):
        # train_dataset = CompSegDataset(
        train_dataset = HairDataset(
            root_dir=self.data_dir,
            fold_idx=self.fold_idx,
            stage="train",
            image_size=args.image_size,
            augmentation=get_training_augmentation(crop_size=self.crop_size),
            preprocessing=self.preprocessing,
        )

        print(f"Len train set: {len(train_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )
        return train_loader

    def get_val_loader(self):
        # val_dataset = CompSegDataset(
        val_dataset = HairDataset(
            root_dir=self.data_dir,
            fold_idx=self.fold_idx,
            stage="val",
            image_size=args.image_size,
            augmentation=None,
            preprocessing=self.preprocessing,
        )

        print(f"Len val set: {len(val_dataset)}")

        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4
        )
        return val_loader

    def get_model(self):
        if self.args.arch == "unet":
            model = smp.Unet(
                encoder_name=self.args.encoder,
                encoder_weights=ENCODER_WEIGHTS,
                classes=1,
                activation=ACTIVATION,
            )
        else:
            raise NotImplementedError

        return model
    
    def run(self):
        args = self.args
        print(json.dumps(args.__dict__, sort_keys=True, indent=2))

        loss = smp.utils.losses.DiceLoss()

        metrics = [smp.utils.metrics.IoU(threshold=0.5),]

        model = self.get_model()

        optimizer = optim.Adam([dict(
            params=model.parameters(),
            lr=args.lr
        ),])
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=0.1,
            verbose=True
        ) 
        
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
        for epoch_idx in range(0, args.num_epochs):
            print("\nEpoch: {}".format(epoch_idx))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            print(f"Consume {datetime.datetime.now() - st}")
            # do something (save model, change lr, etc.)
            if max_score < valid_logs["iou_score"]:
                max_score = valid_logs["iou_score"]
                torch.save(model, f"./kane_weight/{args.arch}_{args.encoder}_fold_{args.fold_idx}.pth")
                print("Model saved!")

            lr_scheduler.step()

            # self.writer.add_scalars("train", train_logs, epoch_idx)
            # self.writer.add_scalars("val", valid_logs, epoch_idx)
            self.writer.add_scalars(
                "dice",
                {
                    "train": train_logs["dice_loss"],
                    "val": valid_logs["dice_loss"]
                },
                epoch_idx
            )
            self.writer.add_scalars(
                "iou",
                {
                    "train": train_logs["iou_score"],
                    "val": valid_logs["iou_score"]
                },
                epoch_idx
            )



def main():
    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
