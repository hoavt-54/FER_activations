#
#	@author Hoa Vu
#	@email hoavutrongvn at gmail.com
#	@create date 2021-06-12 09:11:29
#	@modify date 2021-06-12 09:11:29
#	@desc [description]
# ======================================================================

import csv
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

#import transforms as transforms

# some constants
cut_size = 44
origin_size = 48


class FER2013(data.Dataset):
    """FER2013 Dataset
    """

    def __init__(self, image_data, targets, transform=None):
        assert (len(image_data) == len(targets)), "Invalid dataset"
        self.transform = transform
        self.data = np.asarray(image_data)
        self.targets = np.asarray(targets)

    
    def __getitem__(self, index):
        img = self.data[index]
        img = np.asarray(img, dtype=np.uint8)
        img = np.reshape(img, (origin_size, origin_size))
        img = np.stack((img,)*3, axis=-1) # img shape:  (48, 48, 3)
        #print("img shape: ", np.shape(img))
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index] # torch.Size([3, 44, 44]), target:  ()

    def __len__(self):
        return len(self.data)


def read_split(data_path=""):
    """ read fer2013.csv file 3 data stream
        Each row has class, img_data, data_set
    """
    image_data = defaultdict(list)
    targets = defaultdict(list)
    with open(data_path) as f:
        next(f)
        for idx, row in enumerate(csv.reader(f)):
            data_set = row[-1]
            img_data = [int(i) for i in row[1].split()]
            image_data[data_set].append(img_data)
            targets[data_set].append(int(row[0]))
            if idx > 100: break

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(44),
        transforms.RandomHorizontalFlip(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.TenCrop(cut_size),
        transforms.Lambda(lambda crops: torch.stack(crops)),
    ])
    
    streams = []
    tfs = [transform_train, transform_test, transform_test]
    for i, dataset in enumerate(["Training", "Training", "Training"]):
        streams.append(FER2013(image_data[dataset], targets[dataset], tfs[i]))

    return streams

    