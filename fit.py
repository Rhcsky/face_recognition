import torch
from torch.nn.utils import clip_grad_norm_

from utils.train_utils import AverageMeter


class DoubleRelationFit:
    def __init__(self, args):
        self.num_class = args.n_way
        self.num_support = args.k_shot
        self.num_query_tr = args.num_query_tr
        self.num_query_val = args.num_query_val

        self.device = args.device

    def train(self, train_loader, model, optimizer, criterion):
        losses = AverageMeter()
        accuracies = AverageMeter()

        self.model_status_train(model)
        for i, data in enumerate(train_loader):
            x, _y = data[0].to(self.device), data[1].to(self.device)

            y_pred, y_query = model(x, _y)

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
    def validate(self, val_loader, model, criterion):
        losses = AverageMeter()
        accuracies = AverageMeter()

        self.model_status_eval(model)
        for i, data in enumerate(val_loader):
            x, _y = data[0].to(self.device), data[1].to(self.device)

            y_pred, y_query = model(x, _y)

            y_one_hot = torch.zeros(self.num_query_val * self.num_class, self.num_class).to(self.device).scatter_(1,
                                                                                                                  y_query.unsqueeze(
                                                                                                                      1),
                                                                                                                  1)
            loss = criterion(y_pred, y_one_hot)

            losses.update(loss.item(), y_pred.size(0))

            y_pred = y_pred.argmax(1)

            accuracy = y_pred.eq(y_query).float().mean()
            accuracies.update(accuracy)

        return losses.avg, accuracies.avg

    def model_status_train(self, model):
        if isinstance(model, torch.nn.DataParallel):
            model.module.custom_train()
        else:
            model.custom_train()
        model.train()

    def model_status_eval(self, model):
        if isinstance(model, torch.nn.DataParallel):
            model.module.custom_eval()
        else:
            model.custom_eval()
        model.eval()
