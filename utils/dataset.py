import os
from glob import glob

import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm

BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_data_dir():
    return os.path.join(BASEDIR, 'data')


class FaceDataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        self.root_dir = BASEDIR + '/data/face/image'
        self.split_description = BASEDIR + '/data/face'

        # self.transform = A.Compose([
        #     A.Normalize(mean=0.441875, std=0.1859375),
        #     ToTensorV2(),
        # ])

        self.transform = transforms.Compose([
            transforms.Resize((112, 92)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.441875, std=0.1859375)
        ])

        self.x, self.y = self.make_dataset(mode)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def make_dataset(self, mode):
        x = []
        y = []

        with open(os.path.join(self.split_description, mode + '.txt'), 'r') as f:
            classes = f.read().splitlines()

        for idx, c in enumerate(tqdm(classes, desc="Making dataset")):
            for img_dir in glob(os.path.join(self.root_dir, c, '*')):
                img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.transform(image=img)['image']
                x.append(img)
                y.append(idx)

        y = torch.LongTensor(y)

        return x, y
