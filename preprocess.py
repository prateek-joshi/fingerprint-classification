import glob
import os
import cv2 as cv
import numpy as np

DATA_DIR = os.path.join('data','NISTSpecialDatabase4GrayScaleImagesofFIGS','NISTSpecialDatabase4GrayScaleImagesofFIGS','sd04','png_txt')
BASE_DIR = os.path.join('data','fingerprints')

def get_class_name(file, folder):
    filepath = os.path.join(folder, file+'.txt')
    with open(filepath, 'r') as f:
        class_name_line = f.readlines()[1].rstrip()
        class_name = class_name_line[-1]
    
    return class_name

def segregate_into_classes(png_files):
    filenames = []
    for file in png_files:
        filename = os.path.split(file)[-1]
        folder = os.path.split(file)[-2]
        filenames.append(filename.split('.')[0])

    for file in filenames:
        class_name = get_class_name(file, folder)

        # Create class directory if not exists
        if not os.path.exists(os.path.join(BASE_DIR,class_name)):
            os.mkdir(os.path.join(BASE_DIR,class_name))

        os.rename(os.path.join(folder,file+'.png'), os.path.join(BASE_DIR,class_name,file+'.png'))


# create base directory if not exists
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

    segregate_into_classes(png_files)