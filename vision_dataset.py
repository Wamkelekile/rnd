
from __future__ import print_function, division
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageFile

MIN_IMG_SIZE = 50

class VisionDataset(Dataset):
    """Vision dataset."""


    def __init__(self, categories, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.categories = sorted(categories)
        self.mapping = {cat: idx for idx, cat in enumerate(self.categories)}
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.all_images = []
        print('Start loading images from dataset %s' % root_dir)
        for cat in categories:
            self.all_images.extend([img for img in (self.root_dir / cat).iterdir() if img.is_file()])
        print('Find %d images belonging to %d classes' % (len(self.all_images), len(self.categories)))

    def __len__(self):
        return len(self.all_images)


    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_label = self.mapping[image_name.parent.name]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(image_name).convert('RGB')
        image_size = image.size
        if image_size[0] < MIN_IMG_SIZE or image_size[1] < MIN_IMG_SIZE:
            image = image.resize((MIN_IMG_SIZE, MIN_IMG_SIZE), Image.ANTIALIAS)
        
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': image_label}
        return sample

