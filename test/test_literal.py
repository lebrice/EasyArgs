import enum
import sys
from dataclasses import dataclass
from typing import Any, List, Literal, NamedTuple, Optional, Union

import pytest

from .testutils import (
    TestSetup,
    exits_and_writes_to_stderr,
    raises_invalid_choice,
    raises_missing_required_arg,
    xfail_param,
)


class FieldComponents(NamedTuple):
    field_annotation: Any
    passed_value: Any
    parsed_value: Any
    incorrect_value: Any


class Color(enum.Enum):
    RED = enum.auto()
    BLUE = enum.auto()
    GREEN = enum.auto()


Fingers = Literal[0, 1, 2, 3, 4, 5]


@pytest.fixture(
    params=[
        FieldComponents(Literal["bob", "alice"], "bob", "bob", "clarice"),
        FieldComponents(Literal[True, False], "True", True, "bob"),
        xfail_param(
            [FieldComponents(Literal[True, False], "true", True, "bob")],
            reason="The support for boolean literals currently assumes just 'True' and 'False'.",
        ),
        FieldComponents(Literal[1, 2, 3], "1", 1, "foobar"),
        FieldComponents(Literal[1, 2, 3], "2", 2, "9"),
        FieldComponents(Literal["bob", "alice"], "bob", "bob", "clarice"),
        FieldComponents(Literal[Color.BLUE, Color.GREEN], "BLUE", Color.BLUE, "red"),
        FieldComponents(Literal[Color.BLUE, Color.GREEN], "BLUE", Color.BLUE, "foobar"),
        FieldComponents(Fingers, "1", 1, "foobar"),
        FieldComponents("Fingers", "1", 1, "foobar"),
    ]
)
def literal_field(request: pytest.FixtureRequest):
    field = request.param  # type: ignore
    return field


def test_literal(literal_field: FieldComponents):
    field_annotation, passed_value, parsed_value, incorrect_value = literal_field

    @dataclass
    class Foo(TestSetup):
        bar: field_annotation  # type: ignore

    with raises_missing_required_arg():
        Foo.setup("")

    assert Foo.setup(f"--bar {passed_value}") == Foo(bar=parsed_value)

    with raises_invalid_choice():
        assert Foo.setup(f"--bar {incorrect_value}")


@pytest.mark.xfail(reason="TODO: add support for optional literals")
def test_optional_literal(literal_field: FieldComponents):
    field_annotation, passed_value, parsed_value, incorrect_value = literal_field

    @dataclass
    class Foo(TestSetup):
        bar: Optional[field_annotation] = None  # type: ignore

    assert Foo.setup("") == Foo(bar=None)
    assert Foo.setup(f"--bar {passed_value}") == Foo(bar=parsed_value)

    with raises_invalid_choice():
        assert Foo.setup(f"--bar {incorrect_value}")


@pytest.mark.xfail(reason="TODO: Support lists of literals.")
def test_list_of_literal(literal_field: FieldComponents):
    field_annotation, passed_value, parsed_value, incorrect_value = literal_field

    @dataclass
    class Foo(TestSetup):
        values: List[field_annotation]  # type: ignore

    with raises_missing_required_arg():
        Foo.setup("")

    assert Foo.setup(f"--values {passed_value} {passed_value}") == Foo(
        values=[parsed_value, parsed_value]
    )
    with raises_invalid_choice():
        assert Foo.setup(f"--values {incorrect_value}")


@dataclass
class SomeFoo(TestSetup):
    param: Literal["bar", "biz"] = "biz"


@pytest.mark.skipif(sys.version_info != (3, 9), reason="Bug is only in 3.9")
@pytest.mark.xfail(strict=True, reason="This bug was fixed by #260")
def test_reproduce_issue_259_parsing_literal_py39():
    """Reproduces https://github.com/lebrice/SimpleParsing/issues/259."""
    # $ python issue.py
    # usage: issue.py [-h] [--param typing.Literal['bar', 'biz']]
    # issue.py: error: argument --param: invalid typing.Literal['bar', 'biz'] value: 'biz'
    with exits_and_writes_to_stderr(
        "argument --param: invalid typing.Literal['bar', 'biz'] value: 'biz'"
    ):
        assert SomeFoo.setup("").param == "biz"


@dataclass
class Foo:
    bar: Union[Literal["a"], int] = "a"


def test_issue_322():
    """Test for https://github.com/lebrice/SimpleParsing/issues/322."""
    from simple_parsing import parse

    assert parse(Foo, args="") == Foo()
    assert parse(Foo, args="--bar=a") == Foo(bar="a")
    with raises_invalid_choice():
        assert parse(Foo, args="--bar=b")

    assert parse(Foo, args="--bar=123") == Foo(bar=123)
    # TODO: What about this again?
    assert parse(Foo, args="--bar=1.23") == Foo(bar=1)
