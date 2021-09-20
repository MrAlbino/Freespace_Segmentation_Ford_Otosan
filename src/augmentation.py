# author: ayberk kose as mralbino

#this file stands for changing image contrasts,color etc.

import numpy as np
import cv2
import os
import tqdm 
from torchvision import transforms as T
from PIL import Image

SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')

#The path to the masks folder is assigned to the variable
image_path=[] #empty list created
for name in os.listdir(IMAGE_DIR):
    image_path.append(os.path.join(IMAGE_DIR,name))
mask_path=[] #empty list created

image_path.sort()

for name in os.listdir(MASK_DIR):
    mask_path.append(os.path.join(MASK_DIR,name))
mask_path.sort()

valid_size = 0.1
test_size  = 0.04
indices = np.random.permutation(len(image_path))
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)
train_input_path_list = image_path[valid_ind:]#We got the elements of the image_path_list list from 1905 to the last element
train_label_path_list = mask_path[valid_ind:]#We got the elements of the mask_path_list list from 1905 to the last element

for image in tqdm.tqdm(train_input_path_list):
    img=Image.open(image)
    color_aug = T.ColorJitter(brightness=0.5, contrast=0.5, hue=0.04)
    img_aug = color_aug(img)
    new_path=image[:-4]+"-1"+".jpg"
    new_path=new_path.replace('images', 'aug_photo')
    img_aug=np.array(img_aug)
    cv2.imwrite(new_path,img_aug)

for mask in tqdm.tqdm(train_label_path_list):
    old_mask=Image.open(mask)
    new_mask=old_mask
    newm_path=mask[:-4]+"-1"+".png"
    newm_path=newm_path.replace('masks', 'aug_masks')
    new_mask.save(newm_path)

    
    