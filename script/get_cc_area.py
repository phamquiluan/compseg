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

from rich import print
from tqdm import tqdm


with Manager() as manager:
    output_list = manager.list()

    def process_single(mask_path):
        mask = cv2.imread(mask_path, 0)
        mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)[1]
        cnts, hiers = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        output_list.append(len(cnts))

    with Pool(int(os.cpu_count() * 0.8)) as p:
        p.map(process_single, glob.glob("./data1/train/mask/*"))

    print(list(set(output_list)))
