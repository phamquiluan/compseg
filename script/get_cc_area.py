"""
get cc area list of the whole dataset with multiprocessing
"""
import argparse
import os
import glob
import cv2
import numpy as np
from multiprocessing.pool import Pool
from multiprocessing import Manager

from table.utils import ensure_color
from rich import print
from tqdm import tqdm


with Manager() as manager:
    w_list = manager.list()
    h_list = manager.list()
    a_list = manager.list()

    def process_single(mask_path):
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (512, 512))
        mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)[1]
        cnts, hiers = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        if "02462" in mask_path:
            return 
        
        mask_name = os.path.basename(mask_path)
        file_num = mask_name.split("_")[0]

        image = None
        overlay = None
        
        write = False
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            w_list.append(w)
            h_list.append(h)
            a_list.append(area)

            if area > 140000:
                if image is None:
                    image = cv2.imread(f"./data1/train/image/{file_num}_in.jpg")
                    image = cv2.resize(image, (512, 512))
                    overlay = image.copy()

                write = True
                cv2.drawContours(overlay, cnt, -1, (0, 255, 0), -1)
    
        if write is True:
            cv2.imwrite(
                f"./debug/{mask_name}",
                np.concatenate((image // 2 + overlay // 2, ensure_color(mask)), axis=1)
            )

    with Pool(int(os.cpu_count() * 0.9)) as p:
        p.map(process_single, list(glob.glob("./data1/train/mask/*")))

    print("w min", min(w_list))
    print("w max", max(w_list))
    print("w avg", int(np.sum(w_list) / len(w_list)))
    print(" ")
    print("h min", min(h_list))
    print("h max", max(h_list))
    print("h avg", int(np.sum(h_list) / len(h_list)))
    print(" ")
    print("area min", int(min(a_list)))
    print("area max", int(max(a_list)))
    print("area avg", int(np.sum(a_list) / len(a_list)))
