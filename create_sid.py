import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from table.utils import show, ensure_color, ensure_gray, overlay_mask

from multiprocessing import Manager

image_size = 1024
count = 0

for image_path in glob.glob("./data/train/image/*"):
# def process_single(image_path):
    image_name = os.path.basename(image_path)
    file_name = os.path.splitext(image_name)[0]
    file_num = image_name.split("_")[0]

    image = cv2.imread(image_path)
        
    ed_mask_path = f"./data/train/mask/{file_num}_ed_gt.png"
    sn_mask_path = f"./data/train/mask/{file_num}_sn_gt.png"
        
    ed_mask = cv2.imread(ed_mask_path, 0)
    sn_mask = cv2.imread(sn_mask_path, 0)

    ed_mask = cv2.threshold(ed_mask, 127.5, 255, cv2.THRESH_BINARY)[1]
    sn_mask = cv2.threshold(sn_mask, 127.5, 255, cv2.THRESH_BINARY)[1]
        
    # get all boundary, cut image, cut hw :))
    cnts = cv2.findContours(ed_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
    for cnt in cnts:

        x, y, w, h = cv2.boundingRect(cnt)
        
        sub_image = cv2.resize(image[y:y + h, x:x + w], (image_size, image_size))
        sub_mask = cv2.resize(sn_mask[y:y + h, x:x + w], (image_size, image_size))

        if np.sum(sub_mask) > 0:
            count += 1

            cv2.imwrite(f"./sid/image/{count}.jpg", sub_image)
            cv2.imwrite(f"./sid/mask/{count}.png", sub_mask)

            cv2.imwrite(
                f"./debug/{count}.jpg",
                np.concatenate((overlay_mask(sub_image, sub_mask), ensure_color(sub_mask)), axis=1)
            )

# from multiprocessing.pool import Pool
# 
# with Pool(int(os.cpu_count() * 0.8)) as p:
#     p.map(process_single, glob.glob("./data/train/image/*"))
