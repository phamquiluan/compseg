import os
import glob
import random
import shutil

assert os.path.exists("train_dataset")


for image_path in glob.glob("train_dataset/input/*"):
	image_name = os.path.basename(image_path)
	file_num = os.path.splitext(image_name)[0].split("_")[0]

	mask_name = f"{file_num}_ed_gt.png"
	mask_path = f"train_dataset/edge_segmentation_gt/{mask_name}"

	rand_val = random.random() 

	if rand_val < 0.15:
		# test 
		shutil.copy(image_path, f"data1/test/image/{image_name}")
		shutil.copy(mask_path, f"data1/test/mask/{mask_name}")
	elif 0.15 < rand_val < 0.3:
		# val
		shutil.copy(image_path, f"data1/val/image/{image_name}")
		shutil.copy(mask_path, f"data1/val/mask/{mask_name}")
	else:
		# train
		shutil.copy(image_path, f"data1/train/image/{image_name}")
		shutil.copy(mask_path, f"data1/train/mask/{mask_name}")
