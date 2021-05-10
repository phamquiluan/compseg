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

    mtta.cuda()
    mtta.eval()

    image_size = 1024

    print(f"IMAGE SIZE: {image_size}")

    for image_path in tqdm(glob.glob("./test_set/input/*")):
        image_name = os.path.basename(image_path)
        file_name = os.path.splitext(image_name)[0]
        file_num = image_name.split("_")[0]

        # read 
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # load.. task 1
        mask_task_1 = cv2.imread(f"./ar_task1/task1_output/final/{file_num}_in.png", 0)
        mask_task_1 = cv2.resize(mask_task_1, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_task_1 = cv2.threshold(mask_task_1, 127.5, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(mask_task_1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
        
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for idx, cnt in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(cnt)

            sub_image = image[y:y + h, x:x + w]

            sub_image = cv2.resize(sub_image, (image_size, image_size))
                
            timage = preprocessing(sub_image)
            tensor = torch.FloatTensor(to_tensor(timage)[None, ...])
            tensor = tensor.cuda()

            with torch.no_grad():
                mask_tta = mtta(tensor)
                mask_tta = (mask_tta.cpu().numpy() * 255).astype(np.uint8)[0][0]
                mask_tta = cv2.threshold(mask_tta, 127.5, 255, cv2.THRESH_BINARY)[1]

                # cv2.imwrite(
                #     os.path.join(args.visualize_path, f"{file_name}_{idx}.png"),
                #     np.concatenate((overlay_mask(sub_image, mask_tta), ensure_color(mask_tta)), axis=1)
                # )
               
                mask_tta = cv2.resize(mask_tta, (w, h))
                mask[y:y + h, x:x + w] = np.maximum(mask[y:y + h, x:x + w], mask_tta)

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
