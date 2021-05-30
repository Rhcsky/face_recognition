from attr import dataclass


@dataclass
class DoubleRelationConfig:
    architecture: str = 'double_relation'
    conv_dim: int = 64
    fc_dim: int = 256
