import os
import json
import cv2
import numpy as np
import albumentations as albu
from torch.utils.data import Dataset

def get_training_augmentation(crop_size=None):

    if crop_size is None:
        crop_size = 320

    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        # hmm, de y con so 320 nay :))
        albu.PadIfNeeded(
            min_height=crop_size, min_width=crop_size, always_apply=True, border_mode=0
        ),
        albu.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
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


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


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


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class CompSegDataset(Dataset):
    def __init__(
        self,
        root_dir,
        fold_idx,
        stage,
        image_size,
        augmentation=None,
        preprocessing=None,
    ):
        assert image_size % 32 == 0, image_size
        self.image_size = image_size

        self.image_dir = os.path.join(root_dir, "image")
        self.mask_dir = os.path.join(root_dir, "mask")
        fold_info_path = os.path.join(root_dir, "fold_info.json")
        
        assert stage in ["train", "val", "test"]
        assert os.path.exists(root_dir)
        assert os.path.exists(self.image_dir)
        assert os.path.exists(self.mask_dir)
        assert os.path.exists(fold_info_path)
       
        self.stage = stage
        self.fold_idx = str(fold_idx)
        self.image_name_list = [] 

        with open(fold_info_path) as ref:
            fold_info = json.load(ref)

            for _idx, _image_name_list in fold_info.items():
                if self.stage in ["val", "test"] and _idx == self.fold_idx:
                    self.image_name_list.extend(_image_name_list)

                if self.stage == "train" and _idx != self.fold_idx:
                    self.image_name_list.extend(_image_name_list)
        
        assert len(self.image_name_list) > 0

        # # for debug
        # if stage == "train":
        #     self.image_name_list = self.image_name_list[:100]
        # else:
        #     self.image_name_list = self.image_name_list[:10]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing


    def __getitem__(self, i):
        image_name = self.image_name_list[i]
        
        # TODO: can than cho nay
        # image_path = os.path.join(self.image_dir, f"{image_name}_in.jpg")
        image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        assert os.path.exists(image_path), image_path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # TODO: can than cho nay
        # mask_path = os.path.join(self.mask_dir, f"{image_name}_sn_gt.png")
        mask_path = os.path.join(self.mask_dir, f"{image_name}.png")
        assert os.path.exists(mask_path), mask_path
        mask = cv2.imread(mask_path, 0)
        
        if "sn_gt" in mask_path:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((50, 50)))            

        # for test :))
        image_size = self.image_size
        image = cv2.resize(image, (image_size, image_size))
        mask = cv2.resize(mask, (image_size, image_size))

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
        
        # print(image.shape, mask.shape)
        return image, mask

    def __len__(self):
        return len(self.image_name_list)


class HairDataset(Dataset):
    def __init__(
        self,
        root_dir,
        fold_idx,
        stage,
        image_size,
        augmentation=None,
        preprocessing=None,
    ):
        assert image_size % 32 == 0, image_size
        self.image_size = image_size

        self.image_dir = os.path.join(root_dir, "image")
        self.mask_dir = os.path.join(root_dir, "mask")
        fold_info_path = os.path.join(root_dir, "fold_info.json")
        
        assert stage in ["train", "val", "test"]
        assert os.path.exists(root_dir)
        assert os.path.exists(self.image_dir)
        assert os.path.exists(self.mask_dir)
        assert os.path.exists(fold_info_path)
       
        self.stage = stage
        self.fold_idx = str(fold_idx)
        self.image_name_list = [] 

        with open(fold_info_path) as ref:
            fold_info = json.load(ref)

            for _idx, _image_name_list in fold_info.items():
                if self.stage in ["val", "test"] and _idx == self.fold_idx:
                    self.image_name_list.extend(_image_name_list)

                if self.stage == "train" and _idx != self.fold_idx:
                    self.image_name_list.extend(_image_name_list)
        
        assert len(self.image_name_list) > 0

        # # for debug
        # if stage == "train":
        #     self.image_name_list = self.image_name_list[:100]
        # else:
        #     self.image_name_list = self.image_name_list[:10]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing


    def __getitem__(self, i):
        image_name = self.image_name_list[i]
        
        # TODO: can than cho nay
        # image_path = os.path.join(self.image_dir, f"{image_name}_in.jpg")
        image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        assert os.path.exists(image_path), image_path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # TODO: can than cho nay
        # mask_path = os.path.join(self.mask_dir, f"{image_name}_sn_gt.png")
        mask_path = os.path.join(self.mask_dir, f"{image_name}.png")
        assert os.path.exists(mask_path), mask_path
        mask = cv2.imread(mask_path)
        mask = mask[...,2] 

        # for test :))
        image_size = self.image_size
        image = cv2.resize(image, (image_size, image_size))
        mask = cv2.resize(mask, (image_size, image_size))

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
        return len(self.image_name_list)


if __name__ == "__main__":

    dataset = HairDataset(
        root_dir="/home/luan/research/compseg/data/kane/",
        fold_idx=1,
        stage="train",
        image_size=512
    )
    
    # from table.utils import ensure_color
    # for idx, (image, mask) in enumerate(dataset):
    #     if idx == 100:
    #         break

    #     cv2.imwrite(f"./debug/{idx}.png", np.concatenate((image, ensure_color((mask * 255).astype(np.uint8))), axis=1))
