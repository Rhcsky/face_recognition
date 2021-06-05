import os
import cv2

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
import face_recognition

from fit import DoubleRelationFit2


def crop_face(img):
    location = face_recognition.face_locations(img, model='hog')  # top, right, bottom, left
    location = location[0]

    top, right, bottom, left = location
    img = img[top:bottom, left:right]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def get_model(fvname='face_vector'):
    facebank = torch.load(os.path.join(f'facebank/{fvname}.pt'))
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
    image = cv2.imread(image)
    image = crop_face(image)
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
    img_path = "some_path.jpg"
    best_prob, idx, ans = inference(img_path, *args)
