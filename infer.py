import os

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

from fit import DoubleRelationFit2


def get_model(fname='face_vector'):
    facebank = torch.load(os.path.join(f'facebank/{fname}.pt'))
    with open(os.path.join('facebank/classes.txt'), 'r') as f:
        classes = f.read().splitlines()

    embedding = torch.load(os.path.join('saved_model/embedding.pt')).to('cuda')
    model = torch.load(os.path.join('saved_model/double_relation.pt')).to('cuda')
    model.n_way = 5
    model.k_support = 1
    model.k_query = 1
    model.k_query_val = 1
    run = DoubleRelationFit2(5, 1, 1, 1, 'cuda')

    transform = A.Compose([
        A.Resize(92, 92),
        A.Normalize(mean=0.441875, std=0.1859375),
        ToTensorV2(),
    ])

    return [run, model, embedding, transform, facebank, classes]


def inference(image, run, model, embedding, transform, facebank, classes):
    # img_dir = os.path.join('capture/images/s1/1.png')
    # image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = transform(image=image)['image'].unsqueeze(0).float().to('cuda')
    best_prob, idx = list(map(lambda x: x.item(), run.infer(facebank, image, model, embedding)))

    if best_prob > -5:
        print(classes)
        print(f"Answer : {classes[idx]}")
        return best_prob, idx, classes[idx]
    else:
        return best_prob, idx, ""


if __name__ == '__main__':
    args = get_model()
    img = "some_path.jpg"
    best_prob, idx, ans = inference(img, *args)
