import os
import glob
import cv2
import numpy as np
import albumentations as albu

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import segmentation_models_pytorch as smp

from table.utils import ensure_color


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        albu.PadIfNeeded(
            min_height=320, min_width=320, always_apply=True, border_mode=0
        ),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.PadIfNeeded(384, 480)]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def ensure_div_32(image):
    s0 = image.shape[0]
    s1 = image.shape[1]
    if s0 % 32 != 0:
        s0 = s0 // 32
        s0 = s0 * 32

    if s1 % 32 != 0:
        s1 = s1 // 32
        s1 = s1 * 32
        
    image = cv2.resize(image, (s1, s0))
    return image


class CompSegDataset(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = [
            os.path.splitext(f)[0].split("_")[0] for f in os.listdir(images_dir)
        ]

        self.images_fps = [
            os.path.join(images_dir, f"{image_id}_in.jpg") for image_id in self.ids
        ]
        self.masks_fps = [
            os.path.join(masks_dir, f"{image_id}_ed_gt.png") for image_id in self.ids
        ]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        assert os.path.exists(self.images_fps[i]), self.images_fps[i]
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ensure_div_32(image)

        assert os.path.exists(self.masks_fps[i]), self.masks_fps[i]
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = ensure_div_32(mask)
        mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)[1] / 255.

        mask = mask.astype("float")
        
        if len(mask.shape) == 2:
            mask = mask[..., None]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


def main():
    dataset = CompSegDataset(
        images_dir="./train_dataset/input/",
        masks_dir="./train_dataset/edge_segmentation_gt/",
        augmentation=get_training_augmentation(),
    )

    ENCODER = "se_resnext50_32x4d"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = (
        "sigmoid"  # could be None for logits or 'softmax2d' for multicalss segmentation
    )
    DEVICE = "cuda"

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    x_train_dir = "./data1/train/image"
    y_train_dir = "./data1/train/mask"
    x_valid_dir = "./data1/val/image"
    y_valid_dir = "./data1/val/mask"

    train_dataset = CompSegDataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = CompSegDataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=0.0001),
        ]
    )

    # it is a simple loop of iterating over dataloader`s samples
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

    # train model for 40 epochs

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


if __name__ == "__main__":
    main()
