import os
import glob
import cv2
import numpy as np


for mask_path in glob.glob("./output_task1/fold1/*"):
    mask_name = os.path.basename(mask_path)

    m1 = cv2.imread(mask_path)
    m2 = cv2.imread(f"./output_task1/fold2/{mask_name}")
    m3 = cv2.imread(f"./output_task1/fold3/{mask_name}")
    m4 = cv2.imread(f"./output_task1/fold4/{mask_name}")
    m5 = cv2.imread(f"./output_task1/fold5/{mask_name}")

    m = m1 / 5 + m2 / 5 + m3 / 5 + m4 / 5 + m5 / 5

    cv2.imwrite(f"./output_task1/merge/{mask_name}", m)
