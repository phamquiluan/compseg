"""
get image shape list of the whole dataset with multiprocessing
"""
import argparse
import os
import glob
import cv2
import numpy as np
from multiprocessing.pool import Pool
from multiprocessing import Manager

from rich import print


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-path", required=True)
    args = parser.parse_args()
    return args


with Manager() as manager:
    output_list = manager.list()

    def process_single(image_path):
        image = cv2.imread(image_path)
        output_list.append(image.shape)

    with Pool(int(os.cpu_count() * 0.8)) as p:
        p.map(process_single, glob.glob("./data1/train/image/*"))

    print(list(set(output_list)))
