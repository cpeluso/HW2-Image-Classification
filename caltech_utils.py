from torchvision.datasets import VisionDataset
from random import shuffle, randint
from PIL import Image
import numpy as np
import os
import os.path
import sys

class CaltechUtils():
    def __init__(self):
        print("Initialize CaltechUtils")
        pass

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        pass

    def get_index(self, annotations, tuple):
        index = 0
        for annotation in annotations:
            if annotation == tuple:
                return index
            else:
                index += 1
        pass
