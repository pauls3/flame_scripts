import os
import pandas as pd
from pathlib import Path
import numpy as np


def get_file_names():
    # Get file names (json and images share same names)
    arr0 = os.listdir('/shared/rtis_lab/data/FLAME/custom_split/train_images')
    arr1 = os.listdir('/shared/rtis_lab/data/FLAME/custom_split/train_masks')
    arr2 = []

    for ii in range(0, len(arr0)):
        arr2.append(arr0[ii] + '\t' + arr1[ii])

    return arr2


def write_files_txt(file_names):
    with open('train.lst', 'w') as file1:
        file1.writelines(line + '\n' for line in file_names)


def main():
    file_names = get_file_names()
    write_files_txt(file_names)


if __name__== "__main__":
  main()