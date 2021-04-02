import os
import json
import cv2
import albumentations as albu
from torch.utils.data import Dataset

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
        root_dir,
        fold_idx,
        stage,
        augmentation=None,
        preprocessing=None,
    ):
        self.image_dir = os.path.join(root_dir, "image")
        self.mask_dir = os.path.join(root_dir, "mask")
        fold_info_path = os.path.join(root_dir, "fold_info.json")
        
        assert stage in ["train", "val"]
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
                if self.stage == "val" and _idx == self.fold_idx:
                    self.image_name_list.extend(_image_name_list)

                if self.stage == "train" and _idx != self.fold_idx:
                    self.image_name_list.extend(_image_name_list)

        assert len(self.image_name_list) > 0
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image_name = self.image_name_list[i]
        
        image_path = os.path.join(self.image_dir, f"{image_name}_in.jpg")
        assert os.path.exists(image_path), image_path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ensure_div_32(image)

        mask_path = os.path.join(self.mask_dir, f"{image_name}_ed_gt.png")
        assert os.path.exists(mask_path), mask_path
        mask = cv2.imread(mask_path, 0)
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
        return len(self.image_name_list)


if __name__ == "__main__":

    for i in range(1, 6):
        # print(i)
        dataset = CompSegDataset(
            root_dir="/home/luan/research/compseg/data1/train/",
            fold_idx=str(i),
            stage="train",
        )

    # print(len(dataset))
    
