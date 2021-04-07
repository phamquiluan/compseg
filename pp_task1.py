import os
import glob
import cv2
import numpy as np
from table.utils import show, ensure_color, ensure_gray


for image_path in glob.glob("./wrong/*"):
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)

    center = image.shape[1] // 2
    mask = image[:, center:]
    image = image[:, :center]

    mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)[1]
    overlay = mask.copy()
    
    cnts, hiers = cv2.findContours(ensure_gray(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    
    # fill small cnts :)
    write = False
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 17000:
            cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), -1)
            write = True

