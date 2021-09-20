# author: ayberk kose as mralbino

#this file is for mirroring images
from PIL import Image, ImageOps
import os
import numpy as np
import tqdm 

SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')


image_path=[] #empty list created
for name in os.listdir(IMAGE_DIR):
    image_path.append(os.path.join(IMAGE_DIR,name))
mask_path=[] #empty list created

image_path.sort()

for name in os.listdir(MASK_DIR):
    mask_path.append(os.path.join(MASK_DIR,name))
mask_path.sort()

valid_size = 0.3
test_size  = 0.1
indices = np.random.permutation(len(image_path))
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)
train_input_path_list = image_path[:10]#We got the elements of the image_path_list list from 1905 to the last element
train_label_path_list = mask_path[:10]#We got the elements of the mask_path_list list from 1905 to the last element

for image in tqdm.tqdm(train_input_path_list):
    im=Image.open(image)
    im_mirror = ImageOps.mirror(im)
    new_path=image[:-4]+"-2"+".jpg"
    new_path=new_path.replace('images', 'aug_photo2')
    im_mirror.save(new_path, quality=95)

for mask in tqdm.tqdm(train_label_path_list):
    old_mask=Image.open(mask)
    new_mask=ImageOps.mirror(old_mask)
    newm_path=mask[:-4]+"-2"+".png"
    newm_path=newm_path.replace('masks', 'aug_masks2')
    new_mask.save(newm_path)