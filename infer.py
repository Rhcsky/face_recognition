import logging
import os

import albumentations as A
import cv2
import hydra
import torch
from albumentations.pytorch import ToTensorV2

from configs import BaseConfig
from fit import DoubleRelationFit

log = logging.getLogger(__name__)


def get_channel(dataset_name):
    if dataset_name in ['face', 'omniglot']:
        in_channel = 1
    elif dataset_name in ['miniimagenet']:
        in_channel = 3
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")
    return in_channel


def check_configure(cfg):
    assert cfg.trainer.dataset.lower() in ['face', 'omniglot', 'miniimagenet']


@hydra.main(config_path='configs', config_name="config")
def main(cfg: BaseConfig) -> None:
    transform = A.Compose([
        A.Normalize(mean=0.441875, std=0.1859375),
        ToTensorV2(),
    ])

    origin_dir = hydra.utils.get_original_cwd()
    img_dir = os.path.join(origin_dir, 'facebank/images/s1/1.png')
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = transform(image=image)['image'].unsqueeze(0).float().to('cuda')

    facebank = torch.load(os.path.join(origin_dir, 'facebank/face_vector.pt'))
    with open(os.path.join(origin_dir, 'facebank/classes.txt'), 'r') as f:
        classes = f.read().splitlines()

    embedding = torch.load(os.path.join(origin_dir, 'saved_model/embedding.pt')).to('cuda')
    model = torch.load(os.path.join(origin_dir, 'saved_model/double_relation.pt')).to('cuda')
    model.n_way = 5
    model.k_support = 1
    model.num_query_tr = 1
    model.num_query_val = 1
    run = DoubleRelationFit(cfg.trainer)

    idx = run.infer(facebank, image, model, embedding)
    print(classes)
    print(f"Answer : {classes[idx]}")


if __name__ == '__main__':
    main()
