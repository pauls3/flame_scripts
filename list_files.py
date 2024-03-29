import os
import pandas as pd
from pathlib import Path
import numpy as np


def get_file_names():
    # Get file names (json and images share same names)
    img_dir = '/shared/rtis_lab/data/FLAME/custom_split/test_images/'
    mask_dir = '/shared/rtis_lab/data/FLAME/custom_split/test_masks/'

    arr0 = os.listdir(img_dir)
    arr1 = os.listdir(mask_dir)
    arr2 = []

    arr0.sort()
    arr1.sort()

    for ii in range(0, len(arr0)):
        arr0[ii] = img_dir + arr0[ii]

    for ii in range(0, len(arr1)):
        arr1[ii] = mask_dir + arr1[ii]

    for ii in range(0, len(arr0)):
        arr2.append(arr0[ii] + '\t' + arr1[ii])

    return arr2


def write_files_txt(file_names):
    with open('test.lst', 'w') as file1:
        file1.writelines(line + '\n' for line in file_names)


def main():
    file_names = get_file_names()
    write_files_txt(file_names)


if __name__== "__main__":
  main()