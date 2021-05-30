from attr import dataclass
from omegaconf import MISSING
import torch


@dataclass
class DoubleRelationConfig:
    architecture: str = 'double_relation'
    conv_dim: int = 64
    fc_dim: int = 256
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
