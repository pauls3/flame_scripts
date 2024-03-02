import json
import os
import pandas as pd
import random
from pathlib import Path
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from typing import Tuple



def get_train_val_test_dataset_split(
    images: np.ndarray, masks: np.ndarray, test_size: float = .15,
    val_size: float = .15, seed: int = 42
    ) -> Tuple[Tuple[np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray]]:
    """Split the images and mask dataset into train, validation and test.

    Parameters
    ----------
    images : ndarray
        The images of the dataset.
    masks : ndarray
        The segmentation masks of the dataset.
    test_size : float, optional
        The test size ratio, by default 0.15.
    val_size : float, optional
        The validation size ratio, by default 0.15.
    seed : int, optional
        The seed to use for the split, by default 42.

    Returns
    -------
    (ndarray, ndarray)
        Tuple containing the input images and the segmentation masks
        of the train set.
    (ndarray, ndarray)
        Tuple containing the input images and the segmentation masks
        of the validation set.
    (ndarray, ndarray)
        Tuple containing the input images and the segmentation masks
        of the test set.
    """
    X_train, X_test, y_train, y_test =  train_test_split(
        images, masks, test_size=test_size, shuffle=True, random_state=seed)
    X_train, X_val, y_train, y_val =  train_test_split(
        X_train, y_train, test_size=val_size, shuffle=True, random_state=seed)

    #return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    return X_train, X_val, X_test


def get_file_names():
    # Get file names (json and images share same names)
    arr0 = os.listdir('Images')
    arr1 = []

    for i in arr0:
        arr1.append(Path(i).stem)

    return arr1


def create_splits(train, val, test):
    print('copying and pasting files...')

    for ii in range(0, len(train)):
        shutil.copy2(src='Images/' + train[ii] + '.jpg', dst='custom_split/train_images/' + train[ii] + '.jpg')
        shutil.copy2(src='Masks/' + train[ii] + '.png', dst='custom_split/train_masks/' + train[ii] + '.png')

    for ii in range(0, len(val)):
        shutil.copy2(src='Images/' + val[ii] + '.jpg', dst='custom_split/validation_images/' + val[ii] + '.jpg')
        shutil.copy2(src='Masks/' + val[ii] + '.png', dst='custom_split/validation_masks/' + val[ii] + '.png')

    for ii in range(0, len(test)):
        shutil.copy2(src='Images/' + test[ii] + '.jpg', dst='custom_split/test_images/' + test[ii] + '.jpg')
        shutil.copy2(src='Masks/' + test[ii] + '.png', dst='custom_split/test_masks/' + test[ii] + '.png')
        
    train.sort()
    test.sort()
    val.sort()
    with open('train.txt', 'w') as file1:
        file1.writelines(line + '\n' for line in train)
    with open('test.txt', 'w') as file1:
        file1.writelines(line + '\n' for line in test)
    with open('val.txt', 'w') as file1:
        file1.writelines(line + '\n' for line in val)
    


def main():
    images = get_file_names()
    train, val, test = get_train_val_test_dataset_split(images, images)
    create_splits(train, val, test)


if __name__== "__main__":
  main()