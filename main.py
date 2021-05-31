import logging

import hydra
import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from configs import BaseConfig
from dataloader import get_dataloader
from models.embedding import Embedding

log = logging.getLogger(__name__)
best_acc = 0


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_model(model_cfg, train_cfg):
    if 'double_relation' == model_cfg.architecture:
        from models.double_relation import DoubleRelationNet
        from fit import DoubleRelationFit

        model = DoubleRelationNet(train_cfg.n_way, train_cfg.k_shot, train_cfg.num_query_tr, train_cfg.num_query_val,
                                  model_cfg.conv_dim, model_cfg.fc_dim).to(train_cfg.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        run = DoubleRelationFit(train_cfg)

        return model, criterion, run
    else:
        raise ValueError


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
    global best_acc
    print(cfg)
    check_configure(cfg)

    init_seed(cfg.trainer.seed)

    train_loader, val_loader = get_dataloader(cfg.trainer, 'train', 'val')
    in_channel = get_channel(cfg.trainer.dataset.lower())

    embedding = Embedding(in_channel).to(cfg.trainer.device)
    model, criterion, run = get_model(cfg.model, cfg.trainer)

    optimizer = torch.optim.Adam(model.parameters(), cfg.trainer.lr)

    cudnn.benchmark = True

    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer,
                                                          lr_lambda=lambda epoch: 0.9)
    writer = SummaryWriter()

    log.info(f"model parameter : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(cfg.trainer.epochs):
        train_loss, train_acc = run.train(train_loader, model, embedding, optimizer, criterion)

        is_test = False if epoch % cfg.trainer.test_iter else True

        if is_test or epoch == cfg.trainer.epoches - 1:
            val_loss, val_acc = run.validate(val_loader, model, embedding, criterion)

            if val_acc >= best_acc:
                is_best = True
                best_acc = val_acc
            else:
                is_best = False

            # if cfg.trainer.save_model:
            # save_checkpoint({
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'best_acc1': best_acc,
            #     'epoch': epoch,
            # }, is_best)

            if is_best:
                torch.save(model, 'double_relation.pt')
                torch.save(embedding, 'embedding.pt')

            train_log = f"[{epoch + 1}/{cfg.trainer.epochs}] {train_loss:.3f}, {val_loss:.3f}, {train_acc:.3f}, {val_acc:.3f}, # {best_acc:.3f}"
            log.info(train_log)

            writer.add_scalar("Acc/Train", train_acc, epoch + 1)
            writer.add_scalar("Acc/Val", val_acc, epoch + 1)
            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Val", val_loss, epoch + 1)

        else:
            train_log = f"[{epoch + 1}/{cfg.trainer.epochs}] {train_loss:.3f}, {train_acc:.3f}"
            log.info(train_log)

        scheduler.step()
    writer.close()


if __name__ == '__main__':
    main()
