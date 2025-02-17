import argparse
from cgi import test
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function
    
    # get files
    
    train_ratio = 0.6
    val_ratio = 0.2
    
    files = sorted(glob.glob(source + '/*.tfrecord'))
    
    for name in ['train', 'val', 'test']:
        
        folder = os.path.join(destination, name)
        os.makedirs(folder, exist_ok=True)

    random.shuffle(files)
    
    train_data = int(len(files) * train_ratio)
    val_data = int(len(files) * val_ratio)
    
    for idx in range(train_data):
        _, name = os.path.split(files[idx])
        ori = os.path.join(source, name)
        ziel = os.path.join(destination, 'train', name)
        shutil.move(ori, ziel)
    
    for idx in range(train_data, train_data + val_data):
        _, name = os.path.split(files[idx])
        ori = os.path.join(source, name)
        ziel = os.path.join(destination, 'val', name)
        shutil.move(ori, ziel)
        
    for idx in range(train_data + val_data, len(files)):
        _, name = os.path.split(files[idx])
        ori = os.path.join(source, name)
        ziel = os.path.join(destination, 'test', name)
        shutil.move(ori, ziel)
        
    logger.info('Split created successfully.')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)
    
    #segment-11236550977973464715_3620_000_3640_000_with_camera_labels.tfrecord