from typing import Any

from attr import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from configs.model import DoubleRelationConfig
from configs.trainer import DoubleRelationTrainer


@dataclass
class BaseConfig:
    model: Any = MISSING
    trainer: Any = MISSING


cs = ConfigStore.instance()
cs.store(name="config_schema", node=BaseConfig)
cs.store(group="model", name="double_relation_schema", node=DoubleRelationConfig)
cs.store(group="trainer", name='double_relation_trainer_schema', node=DoubleRelationTrainer)
