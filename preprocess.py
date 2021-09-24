import glob
import os
import cv2 as cv

DATA_DIR = os.path.join('data','NISTSpecialDatabase4GrayScaleImagesofFIGS','NISTSpecialDatabase4GrayScaleImagesofFIGS','sd04','png_txt')

# create base directory if not exists
BASE_DIR = os.path.join('data','fingerprints')
if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)

# collect all sub-folders
folders_list = []
for root, dirs, files in os.walk(DATA_DIR):
    for dir in dirs:
        folders_list.append(dir)
    break

for folder in folders_list:
    folder_path = os.path.join(DATA_DIR, folder)

    png_files = glob.glob(os.path.join(folder_path,'*.png'))
    txt_files = glob.glob(os.path.join(folder_path,'*.txt'))

    # TODO: function to preprocess image and seperate into folders based on class
    segregate_into_classes(png_files, txt_files)

    break