from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Union

import pytest

from simple_parsing import replace

logger = logging.getLogger(__name__)


@dataclass
class A:
    a: float = 0.0


@dataclass
class B:
    b: str = "bar"
    b_post_init: str = field(init=False)

    def __post_init__(self):
        self.b_post_init = self.b + "_post"


@dataclass
class UnionConfig:
    a_or_b: Union[A, B] = field(default_factory=A)


@dataclass
class Level1:
    level: int = 1
    name: str = "level1"


@dataclass
class Level2:
    level: int = 2
    name: str = "level2"
    prev: Level1 = field(default_factory=Level1)


@dataclass
class Level3:
    level: int = 3
    name: str = "level3"
    prev: Level2 = field(
        default_factory=functools.partial(Level2, name="level2_foo"))


@dataclass
class InnerPostInit:
    in_arg: float = 1.0
    in_arg_post: str = field(init=False)
    for_outer_post: str = "foo"

    def __post_init__(self):
        self.in_arg_post = str(self.in_arg)


@dataclass
class OuterPostInit:
    out_arg: int = 1
    out_arg_post: str = field(init=False)
    inner: InnerPostInit = field(default_factory=InnerPostInit)
    arg_post_on_inner: str = field(init=False)

    def __post_init__(self):
        self.out_arg_post = str(self.out_arg)
        self.arg_post_on_inner = self.inner.for_outer_post + "_outer"


@pytest.mark.parametrize(
    ("dest_config", "src_config", "changes_dict"),
    [
        (A(a=2.0), A(), {"a": 2.0}),
        (A(), A(a=2.0), {"a": 0.0}),
        (B(b="test"), B(), {"b": "test"}),
        (B(), B(b="test1"), {"b": "bar"}),
    ],
)
def test_replace_plain_dataclass(dest_config: object, src_config: object, changes_dict: dict):
    config_replaced = replace(src_config, changes_dict)
    assert config_replaced == dest_config


@pytest.mark.parametrize(
    ("dest_config", "src_config", "changes_dict"),
    [
        (Level1(name="level1_the_potato"),
         Level1(), {"name": "level1_the_potato"}),
        (Level2(name="level2_bar"), Level2(), {"name": "level2_bar"}),
        (
            Level2(name="level2_bar", prev=Level1(name="level1_good")),
            Level2(),
            {"name": "level2_bar", "prev": {"name": "level1_good"}},
        ),
        (
            Level2(name="level2_bar", prev=Level1(name="level1_good")),
            Level2(),
            {"name": "level2_bar", "prev.name": "level1_good"},
        ),
        (
            Level3(
                name="level3_greatest",
                prev=Level2(name="level2_greater",
                            prev=Level1(name="level1_greate")),
            ),
            Level3(),
            {
                "name": "level3_greatest",
                "prev": {"name": "level2_greater", "prev": {"name": "level1_greate"}},
            },
        ),
        (
            Level3(
                name="level3_greatest",
                prev=Level2(name="level2_greater",
                            prev=Level1(name="level1_greate")),
            ),
            Level3(),
            {
                "name": "level3_greatest",
                "prev.name": "level2_greater",
                "prev.prev.name": "level1_greate",
            },
        ),
    ],
)
def test_replace_nested_dataclasses(dest_config: object, src_config: object, changes_dict: dict):
    config_replaced = replace(src_config, changes_dict)
    assert config_replaced == dest_config


@pytest.mark.parametrize(
    ("dest_config", "src_config", "changes_dict"),
    [
        (InnerPostInit(in_arg=2.0), InnerPostInit(), {"in_arg": 2.0}),
        (
            OuterPostInit(out_arg=2, inner=(
                InnerPostInit(3.0, for_outer_post="bar"))),
            OuterPostInit(),
            {"out_arg": 2, "inner": {"in_arg": 3.0, "for_outer_post": "bar"}},
        ),
        (
            OuterPostInit(out_arg=2, inner=(
                InnerPostInit(3.0, for_outer_post="bar"))),
            OuterPostInit(),
            {"out_arg": 2, "inner.in_arg": 3.0, "inner.for_outer_post": "bar"},
        ),
    ],
)
def test_replace_post_init(dest_config: object, src_config: object, changes_dict: dict):
    config_replaced = replace(src_config, changes_dict)
    assert config_replaced == dest_config


@pytest.mark.parametrize(
    ("dest_config", "src_config", "changes_dict"),
    [
        (
            UnionConfig(a_or_b=B(b='bob')),
            UnionConfig(a_or_b=A(a=1.0)),
            {'a_or_b': B(b='bob')}
        ),
        (
            UnionConfig(a_or_b=A(a=2.0)),
            UnionConfig(a_or_b=A(a=1.0)),
            {'a_or_b.a': 2.0}
        ),
    ]
)
def test_replace_union_dataclass(dest_config: object, src_config: object, changes_dict: dict):
    config_replaced = replace(src_config, changes_dict)
    assert config_replaced == dest_config
