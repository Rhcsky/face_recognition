import hydra
from configs import BaseConfig


@hydra.main(config_path='configs', config_name="config")
def main(cfg: BaseConfig) -> None:
    print(cfg)


if __name__ == '__main__':
    main()
