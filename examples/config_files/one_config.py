"""Example adapted from https://github.com/eladrich/pyrallis#my-first-pyrallis-example-"""
from dataclasses import dataclass

import simple_parsing


@dataclass
class TrainConfig:
    """Training configuration."""

    workers: int = 8  # The number of workers for training
    exp_name: str = "default_exp"  # The experiment name


def main(args=None) -> None:
    cfg = simple_parsing.parse(
        config_class=TrainConfig, args=args, add_config_path_arg="config-file"
    )
    print(f"Training {cfg.exp_name} with {cfg.workers} workers...")


main()
expected = """
Training default_exp with 8 workers...
"""

main("")
expected += """\
Training default_exp with 8 workers...
"""

# NOTE: When running as in the readme:
main("--config-file one_config.yaml --exp_name my_first_exp")
expected += """\
Training my_first_exp with 42 workers...
"""
