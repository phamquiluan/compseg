import os
import glob
import random
import shutil

random.seed(1)

root_dir = "./data/kane"
image_dir = os.path.join(root_dir, "image")
mask_dir = os.path.join(root_dir, "mask")

assert os.path.exists(root_dir)
assert os.path.exists(image_dir)
assert os.path.exists(mask_dir)

file_num_list = []

for image_path in glob.glob(os.path.join(image_dir, "*")):
    image_name = os.path.basename(image_path)
        
    # for kane only
    file_name = os.path.splitext(image_name)[0]
    if not os.path.exists(os.path.join(mask_dir, f"{file_name}.png")):
        continue

    file_num = os.path.splitext(image_name)[0].split("_")[0]
    file_num_list.append(file_num)

# shuffle
# assert len(file_num_list) == 15000, len(file_num_list)

random.shuffle(file_num_list)

# split fold

# fold_info = {
#     "1": file_num_list[:3000],
#     "2": file_num_list[3000:6000],
#     "3": file_num_list[6000:9000],
#     "4": file_num_list[9000:12000],
#     "5": file_num_list[12000:],
# }

fold_info = {
    "1": file_num_list[:711],
    "2": file_num_list[711:711*2],
    "3": file_num_list[711*2:711*3],
    "4": file_num_list[711*3:711*4],
    "5": file_num_list[711*4:],
}


for k, v in fold_info.items():
    print(k, len(v))

import json

with open("data/kane/fold_info.json", "w") as ref:
    json.dump(fold_info, ref)
