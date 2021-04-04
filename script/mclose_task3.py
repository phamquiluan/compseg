import argparse
import os
import glob
from multiprocessing.pool import Pool
from multiprocessing import Manager

import cv2
import numpy as np
from rich import print

from table.utils import overlay_mask


def process_single(image_path):
    image_name = os.path.basename(image_path)
    file_name = os.path.splitext(image_name)[0]
    file_num = file_name.split("_")[0]

    image = cv2.imread(image_path)

    mask_path = f"/data/compseg/signature_segmentation_gt/{file_num}_sn_gt.png"
    assert os.path.exists(mask_path), mask_path

    mask = cv2.imread(mask_path, 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((50, 50)))

    debug_image = overlay_mask(image, mask)
    cv2.imwrite(f"debug/{image_name}", debug_image)


with Pool(int(os.cpu_count() * 0.8)) as p:
    p.map(process_single, list(glob.glob("/data/compseg/input/*"))[:1000])
