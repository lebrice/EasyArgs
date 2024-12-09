"""Tests for config-path option."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from simple_parsing.parsing import ArgumentParser, parse


@dataclass
class BarConf:
    foo: str


@pytest.mark.parametrize(
    "conf_arg_name", ["config-file", "config_file", "foo.bar.baz?", "bob bob bob"]
)
def test_config_path_arg(tmp_path: Path, conf_arg_name: str):
    """Test config_path with valid strings."""
    # Create config file
    conf_path = tmp_path / "foo.yml"
    with conf_path.open("w") as f:
        json.dump({"foo": "bee"}, f)

    # with pytest.raises(ValueError):
    parser = ArgumentParser(BarConf, add_config_path_arg=conf_arg_name)
    args = parser.parse_args([f"--{conf_arg_name}", str(conf_path)])
    print(args)


@pytest.mark.parametrize(
    "conf_arg_name",
    [
        "-------",
    ],
)
def test_pass_invalid_value_to_add_config_path_arg(tmp_path: Path, conf_arg_name: str):
    """Test config_path with invalid strings."""
    # Create config file
    conf_path = tmp_path / "foo.yml"
    with conf_path.open("w") as f:
        json.dump({"foo": "bee"}, f)

    parser = ArgumentParser(BarConf, add_config_path_arg=conf_arg_name)
    with pytest.raises(ValueError):
        parser.parse_args([f"--{conf_arg_name}", str(conf_path)])


def test_config_path_same_as_dst_error():
    """Raise an error if add_config_path_arg and dest are the equal."""
    with pytest.raises(ValueError, match="`add_config_path_arg` cannot be the same as `dest`."):
        parse(BarConf, dest="boo", add_config_path_arg="boo")
