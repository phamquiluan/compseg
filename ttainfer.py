import os
import glob
import argparse

import cv2
import numpy as np
import torch
import ttach as tta
import albumentations as albu
from tqdm import tqdm
from PIL import Image
from table.utils import overlay_mask, ensure_color

import segmentation_models_pytorch as smp
from lab.loader import to_tensor

preprocessing = smp.encoders.get_preprocessing_fn(
    encoder_name="resnet34",
    pretrained="imagenet"
)


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight-path", required=True)
parser.add_argument("-o", "--output-path", required=True)
parser.add_argument("-v", "--visualize-path")
args = parser.parse_args()


@torch.no_grad()
def main():
    tta_trans = tta.Compose([
        tta.VerticalFlip(),
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0,180]),
    ])
        
    m = torch.load(args.weight_path, map_location="cpu")
    mtta = tta.SegmentationTTAWrapper(m, tta_trans,merge_mode="mean")

    m.cuda()
    m.eval()

    mtta.cuda()
    mtta.eval()

    for image_path in tqdm(glob.glob("./test_set/input/*")):
        image_name = os.path.basename(image_path)
        file_name = os.path.splitext(image_name)[0]
    
        # read 
        image_size = 512
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size))
            
        timage = preprocessing(image)
        tensor = torch.FloatTensor(to_tensor(timage)[None, ...])
        tensor = tensor.cuda()

        with torch.no_grad():
            mask_tta = mtta(tensor)
            mask_tta = (mask_tta.cpu().numpy() * 255).astype(np.uint8)[0][0]
            mask_tta = cv2.threshold(mask_tta, 127.5, 255, cv2.THRESH_BINARY)[1]
            debug_image = overlay_mask(image, mask_tta)

            mask = m(tensor)
            mask = (mask.cpu().numpy() * 255).astype(np.uint8)[0][0]
    
        cv2.imwrite(
            os.path.join(args.output_path, f"{file_name}.png"),
            ensure_color(mask)
        )

        if args.visualize_path:
            cv2.imwrite(
                os.path.join(args.visualize_path, f"{file_name}.png"),
                np.concatenate((overlay_mask(image, mask), ensure_color(mask)), axis=1)
            )



if __name__ == "__main__":
    main()
