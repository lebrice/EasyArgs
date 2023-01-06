from dataclasses import dataclass

import pytest

from simple_parsing import helpers
from .testutils import TestSetup


@dataclass
class Base(TestSetup):
    """Some extension of base-class `Base`"""

    a: int = 5
    f: bool = False


@dataclass
class Flags(TestSetup):
    a: bool  # an example required flag (defaults to False)
    b: bool = True  # optional flag 'b'.
    c: bool = False  # optional flag 'c'.


@pytest.mark.parametrize(
    "flag,f",
    [
        ("--a 5", False),
        ("--a 5 --f", True),
        ("--a 5 --nof", False),
        ("--a 5 --f=true", True),
        ("--a 5 --f true", True),
        ("--a 5 --f True", True),
        ("--a 5 --f=false", False),
        ("--a 5 --f false", False),
        ("--a 5 --f False", False),
        ("--a 5 --f --f false", False),
    ],
)
def test_bool_base_work(flag, f):
    ext = Base.setup(flag)
    assert ext.f is f


@pytest.mark.parametrize(
    "flag,a,b,c",
    [
        ("--a", True, True, False),
        ("--a true --b --c", True, True, True),
        ("--a true --noc --b --c --noc", True, True, False),
        ("--noa --b false --noc", False, False, False),
    ],
)
def test_bool_flags_work(flag, a, b, c):
    flags = Flags.setup(flag)
    assert flags.a is a
    assert flags.b is b
    assert flags.c is c


@pytest.mark.parametrize(
    "flag, nargs, a",
    [
        # By default, support both --noflag and --flag=false
        ("--a", None, True),
        ("--noa", None, False),
        ("--a true", None, True),
        ("--a true false", None, SystemExit),
        # 1 argument explicitly required
        ("--a", 1, SystemExit),
        ("--noa", 1, SystemExit),
        ("--a=true", 1, [True]),
        ("--a true false", 1, SystemExit),
        # 2 argument explicitly required
        ("--a", 2, SystemExit),
        ("--noa", 2, SystemExit),
        ("--a=true", 2, SystemExit),
        ("--a true false", 2, [True, False]),
        # 1+ argument explicitly required
        ("--a", '+', SystemExit),
        ("--noa", '+', SystemExit),
        ("--a=true", '+', [True]),
        ("--a true false", '+', [True, False]),
        # 0 or 1+ argument explicitly required
        ("--a", '*', [True]),
        ("--noa", '*', [False]),
        ("--a=true", '*', [True]),
        ("--a true false", '*', [True, False]),
    ],
)
def test_bool_nargs(flag, nargs, a):
    @dataclass
    class MyClass(TestSetup):
        """Some extension of base-class `Base`"""
        a: bool = helpers.field(nargs=nargs)

    if a == SystemExit:
        with pytest.raises(SystemExit):
            MyClass.setup(flag)
    else:
        flags = MyClass.setup(flag)
        assert flags.a == a
