import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.nn.utils import clip_grad_norm_

from utils.common import split_support_query_set
from utils.train_utils import AverageMeter


class DoubleRelationFit:
    def __init__(self, args):
        self.num_class = args.n_way
        self.num_support = args.k_shot
        self.num_query_tr = args.num_query_tr
        self.num_query_val = args.num_query_val

        self.device = args.device

        self.transform = A.Compose([
            ToTensorV2
        ])

    def train(self, train_loader, model, embedding, optimizer, criterion):
        losses = AverageMeter()
        accuracies = AverageMeter()
        num_query = self.num_query_tr

        embedding.train()
        model.train()
        for i, data in enumerate(train_loader):
            x, _y = data[0].to(self.device), data[1].to(self.device)

            _embed = embedding(x)
            support_vector, query_vector, y_support, y_query = split_support_query_set(_embed, _y, self.num_class,
                                                                                       self.num_support,
                                                                                       num_query)

            y_pred = model(support_vector, query_vector)

            y_one_hot = torch.zeros(self.num_query_tr * self.num_class, self.num_class).to(self.device).scatter_(1,
                                                                                                                 y_query.unsqueeze(
                                                                                                                     1),
                                                                                                                 1)
            loss = criterion(y_pred, y_one_hot)

            losses.update(loss.item(), y_pred.size(0))

            y_pred = y_pred.argmax(1)

            accuracy = y_pred.eq(y_query).float().mean()
            accuracies.update(accuracy)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        return losses.avg, accuracies.avg

    @torch.no_grad()
    def validate(self, val_loader, model, embedding, criterion):
        losses = AverageMeter()
        accuracies = AverageMeter()
        num_query = self.num_query_val

        embedding.eval()
        model.eval()
        for i, data in enumerate(val_loader):
            x, _y = data[0].to(self.device), data[1].to(self.device)

            _embed = embedding(x)
            support_vector, query_vector, y_support, y_query = split_support_query_set(_embed, _y, self.num_class,
                                                                                       self.num_support,
                                                                                       num_query)

            y_pred = model(support_vector, query_vector)

            y_one_hot = torch.zeros(self.num_query_tr * self.num_class, self.num_class).to(self.device).scatter_(1,
                                                                                                                 y_query.unsqueeze(
                                                                                                                     1),
                                                                                                                 1)
            loss = criterion(y_pred, y_one_hot)

            losses.update(loss.item(), y_pred.size(0))

            y_pred = y_pred.argmax(1)

            accuracy = y_pred.eq(y_query).float().mean()
            accuracies.update(accuracy)

        return losses.avg, accuracies.avg

    @torch.no_grad()
    def infer(self, facebank, query, model, embedding):
        embedding.eval()
        model.eval()

        query = embedding(query)
        out = model(facebank, query, True)
        print(out)
        return torch.argmax(out)
