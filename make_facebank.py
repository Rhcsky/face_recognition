import os
from glob import glob

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

if __name__ == '__main__':
    embedding = torch.load('saved_model/embedding.pt').to('cuda')
    transform = A.Compose([
        A.Resize(112, 92),
        A.Normalize(mean=0.441875, std=0.1859375),
        ToTensorV2(),
    ])
    classes = glob('facebank/images/*')
    print(classes)

    f = open('facebank/classes.txt', 'w')

    embed_list = list()
    face_list = list()

    for c in tqdm(classes):
        f.write(os.path.basename(c) + '\n')
        embed_list = list()
        for img_dir in glob(os.path.join(c, '*')):
            img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
            img = transform(image=img)['image'].unsqueeze(0).float().to('cuda')
            embed_vector = embedding(img)
            embed_list.append(embed_vector.squeeze(0))

        embed_stack = torch.stack(embed_list, dim=0).mean(0)
        face_list.append(embed_stack)

    face_stack = torch.stack(face_list, dim=0)
    torch.save(face_stack, 'facebank/face_vector.pt')
    f.close()
