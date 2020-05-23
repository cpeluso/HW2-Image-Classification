from torchvision.datasets import VisionDataset
from random import shuffle, randint
from PIL import Image
import numpy as np
import os
import os.path
import sys
from . import CaltechUtils

class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        file = open("./Caltech101/" + split + ".txt", "r")
        annotations = file.readlines()

        # This defines the split you are going to use
        # (split files are called 'train.txt' and 'test.txt')
        self.root = root
        self.utils = CaltechUtils()
        self.split = split 
        self.annotations = [annotation for annotation in annotations if "BACKGROUND" not in annotation]

        classes = list(set([row.split("/")[0] for row in self.annotations]))
        class_index = 0
        class_indices = {}

        for class_string in classes:
          class_indices[class_string] = class_index
          class_index += 1

        self.class_indices = class_indices

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indices to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        pass

    def train_eval_split(self):

        if self.split != "train":
          return _, _

        # Get set of classes
        labels = list(set([row.split("/")[0] for row in self.annotations]))

        train_indices = []
        eval_indices = []

        for label in labels:
          # Get all samples of the current class
          samples = np.array([sample for sample in self.annotations if sample.split("/")[0] == label])
          # Shuffle the data
          shuffle(samples)

          # Assign the same portion of data to train and evaluation set
          length=len(samples)

          if length % 2 != 0:
            train_length = length/2 + randint(0,1)
          else:
            train_length = length/2

          train_data = samples[: int(train_length)]
          eval_data = samples[int(train_length):]

          # Append partial indices
          train_partial_indices = []
          eval_partial_indices = []

          for data in train_data:
            train_partial_indices.append(self.utils.get_index(self.annotations, data))

          for data in eval_data:
            eval_partial_indices.append(self.utils.get_index(self.annotations, data))

          train_indices.append(train_partial_indices)
          eval_indices.append(eval_partial_indices)

        # Flat out the lists of indices
        train_indices = [item for sublist in train_indices for item in sublist]
        eval_indices = [item for sublist in eval_indices for item in sublist]

        return train_indices, eval_indices
        pass

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        img_path = os.path.join(self.root, self.annotations[index].rstrip())
        label = str(self.annotations[index]).split("/")[0]
        image = self.utils.pil_loader(img_path)

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return (image, torch.tensor(self.class_indices[label]))
        pass

    def __len__(self): 
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.annotations)
        return length
        pass
