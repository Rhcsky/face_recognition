import torch
from attr import dataclass


@dataclass
class BasicTrainer:
    seed: int = 7
    worker: int = 8
    dataset: str = "face"
    n_way: int = 5
    k_shot: int = 5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_model: bool = False


@dataclass
class DoubleRelationTrainer(BasicTrainer):
    lr: float = 1e-4
    epochs: int = 30
    test_iter: int = 1
    episode_tr: int = 100
    episode_val: int = 100
    num_query_tr: int = 5
    num_query_val: int = 5
