import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from table.utils import show, ensure_color, ensure_gray


# for mask_path in tqdm(glob.glob("./output_task1/merge/*")):

def process_single(mask_path):
    mask_name = os.path.basename(mask_path)
    file_name = os.path.splitext(mask_name)[0]


    # read and split image?
    # image_path = f"./test_set/input/{file_name}.jpg"
    # assert os.path.exists(image_path), image_path
    # image = cv2.imread(image_path)
    # image = cv2.resize(image, (512, 512))

    mask = cv2.imread(mask_path)
    mask = cv2.threshold(mask, 205, 255, cv2.THRESH_BINARY)[1]
    # overlay = mask.copy()
    mask = ensure_gray(mask)
    
    cnts, hiers = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    
    # fill small cnts :)
    # write = False
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 17000:
            # cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), -1)
            cv2.drawContours(mask, [cnt], -1, 0, -1)
            # write = True
        
    cv2.imwrite(f"./output_task1/final/{mask_name}", mask)
    # if write is True:
    #     cv2.imwrite(f"./debug/{mask_name}", np.concatenate((image,  overlay), axis=1))


from multiprocessing.pool import Pool

with Pool(int(os.cpu_count() * 0.8)) as p:
    p.map(process_single, glob.glob("./output_task1/merge/*"))
