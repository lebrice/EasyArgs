import copy
import dataclasses
import inspect
import itertools
import logging
import math
import pickle
import random
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import Field, InitVar, dataclass, fields
from functools import singledispatch, total_ordering
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
from simple_parsing.helpers.fields import field
from simple_parsing.helpers import Serializable, encode
from simple_parsing.helpers.serialization import register_decoding_fn
from simple_parsing.logging_utils import get_logger
from simple_parsing.utils import (
    compute_identity,
    dict_intersection,
    dict_union,
    field_dict,
    zip_dicts,
)

from .hparam import hparam, log_uniform, uniform
from .priors import LogUniformPrior, NormalPrior, Prior, UniformPrior

logger = get_logger(__file__)
T = TypeVar("T")
HP = TypeVar("HP", bound="HyperParameters")


@dataclass
class BoundInfo(Serializable):
    """Object used to provide the bounds as required by `GPyOpt`."""

    name: str
    # One of 'continuous', 'discrete' or 'bandit' (unsuported).
    type: str = "continuous"
    domain: Tuple[float, float] = (np.NINF, np.Infinity)


@dataclass
class HyperParameters(Serializable, decode_into_subclasses=True):  # type: ignore
    """ Base class for dataclasses of HyperParameters. """

    # Class variable holding the random number generator used to create the
    # samples.
    rng: ClassVar[np.random.RandomState] = np.random

    def __post_init__(self):
        for field in fields(self):
            field: Field
            name = field.name
            value = getattr(self, name)
            # Apply any post-processing function, if applicable.
            if "postprocessing" in field.metadata:
                # print(f"Post-processing of field {name}")
                new_value = field.metadata["postprocessing"](value)
                setattr(self, name, new_value)

    @classmethod
    def field_names(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def id(self):
        return compute_identity(**self.to_dict())

    def seed(self, seed: Optional[int]) -> None:
        """TODO: Seed all priors with the given seed. (recursively if nested dataclasses
        are present.)
        """
        raise NotImplementedError("TODO")

    @classmethod
    def get_priors(cls) -> Dict[str, Prior]:
        """ Returns a dictionary of the Priors for the hparam fields in this class. """
        priors_dict: Dict[str, Prior] = {}
        for field in fields(cls):
            # If a HyperParameters class contains another HyperParameters class as a field
            # we perform returned a flattened dict.
            if inspect.isclass(field.type) and issubclass(field.type, HyperParameters):
                priors_dict[field.name] = field.type.get_priors()
            else:
                prior: Optional[Prior] = field.metadata.get("prior")
                if prior:
                    priors_dict[field.name] = prior
        return priors_dict

    @classmethod
    def get_orion_space_dict(cls) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for field in fields(cls):
            # If a HyperParameters class contains another HyperParameters class as a field
            # we perform returned a flattened dict.
            if inspect.isclass(field.type) and issubclass(field.type, HyperParameters):
                result[field.name] = field.type.get_orion_space_dict()
            else:
                prior: Optional[Prior] = field.metadata.get("prior")
                if prior:
                    result[field.name] = prior.get_orion_space_string()
        return result

    def get_orion_space(self) -> Dict[str, str]:
        """NOTE: This might be more useful in some cases than the above classmethod
        version, for example when a field is a different kind of dataclass than its
        annotation.
        """
        result: Dict[str, str] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            # If a HyperParameters class contains another HyperParameters class as a field
            # we perform returned a flattened dict.
            if isinstance(value, HyperParameters):
                result[field.name] = value.get_orion_space()
            else:
                prior: Optional[Prior] = field.metadata.get("prior")
                if prior:
                    result[field.name] = prior.get_orion_space_string()
        return result

    @classmethod
    def space_id(cls):
        return compute_identity(**cls.get_orion_space_dict())

    @classmethod
    def get_bounds(cls) -> List[BoundInfo]:
        """Returns the bounds of the search domain for this type of HParam.

        Returns them as a list of `BoundInfo` objects, in the format expected by
        GPyOpt.
        """
        bounds: List[BoundInfo] = []
        for f in fields(cls):
            # TODO: handle a hparam which is categorical (i.e. choices)
            min_v = f.metadata.get("min")
            max_v = f.metadata.get("max")
            if min_v is None or max_v is None:
                continue
            if f.type is float:
                bound = BoundInfo(name=f.name, type="continuous", domain=(min_v, max_v))
            elif f.type is int:
                bound = BoundInfo(name=f.name, type="discrete", domain=(min_v, max_v))
            else:
                raise NotImplementedError(
                    f"Unsupported type for field {f.name}: {f.type}"
                )
            bounds.append(bound)
        return bounds

    @classmethod
    def get_bounds_dicts(cls) -> List[Dict[str, Any]]:
        """Returns the bounds of the search space for this type of HParam,
        in the format expected by the `GPyOpt` package.
        """
        return [b.to_dict() for b in cls.get_bounds()]

    @classmethod
    def sample(cls):
        kwargs: Dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            if inspect.isclass(field.type) and issubclass(field.type, HyperParameters):
                # TODO: Should we allow adding a 'prior' in terms of a dataclass field?
                kwargs[field.name] = field.type.sample()
            else:
                prior: Optional[Prior] = field.metadata.get("prior")
                if prior is not None:
                    prior.rng = cls.rng
                    value = prior.sample()
                    if isinstance(value, np.ndarray):
                        value = value.item()
                        assert False, value
                    kwargs[field.name] = value
        return cls(**kwargs)

    def replace(self, **new_params):
        new_hp_dict = dict_union(self.to_dict(), new_params, recurse=True)
        new_hp = type(self).from_dict(new_hp_dict)
        return new_hp

    # @classmethod
    # @contextmanager
    # def use_priors(cls, value: bool = True):
    #     temp = cls.sample_from_priors
    #     cls.sample_from_priors = value
    #     yield
    #     cls.sample_from_priors = temp

    def to_array(self, dtype=np.float32) -> np.ndarray:
        values: List[float] = []
        for k, v in self.to_dict(dict_factory=OrderedDict).items():
            try:
                v = float(v)
            except Exception as e:
                logger.warning(
                    f"Ignoring field {k} because we can't make a float out of it."
                )
            else:
                values.append(v)
        return np.array(values, dtype=dtype)

    @classmethod
    def from_array(cls: Type[HP], array: np.ndarray) -> HP:
        if len(array.shape) == 2 and array.shape[0] == 1:
            array = array[0]

        keys = list(field_dict(cls))
        # idea: could use to_dict and to_array together to determine how many
        # values to get for each field. For now we assume that each field is one
        # variable.
        # cls.sample().to_dict()
        # assert len(keys) == len(array), "assuming that each field is dim 1 for now."
        assert len(keys) == len(array), "assuming that each field is dim 1 for now."
        d = OrderedDict(zip(keys, array))
        logger.debug(f"Creating an instance of {cls} using args {d}")
        d = OrderedDict(
            (k, v.item() if isinstance(v, np.ndarray) else v) for k, v in d.items()
        )
        return cls.from_dict(d)

    def clip_within_bounds(self: HP) -> HP:
        d = self.to_dict()
        for bound in self.get_bounds():
            min_v, max_v = bound.domain
            d[bound.name] = min(max_v, max(min_v, d[bound.name]))
        return self.from_dict(d)


@singledispatch
def save(obj: object, path: Path) -> None:
    """Saves the object `obj` at path `path`.

    Uses pickle at the moment, regardless of the path name or object type.
    TODO: Choose the serialization function depending on the path's extension.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


@save.register
def save_serializable(obj: Serializable, path: Path) -> None:
    obj.save(path)


@total_ordering
class Point(NamedTuple):
    hp: HyperParameters
    perf: float

    def __eq__(self, other: object):
        if not isinstance(other, tuple):
            return NotImplemented
        elif not len(other) == 2:
            return NotImplemented
        # NOTE: This doesn't do any kind
        other_hp, other_perf = other
        hps_equal = self.hp == other_hp
        if not hps_equal and isinstance(other_hp, dict):
            other_id = compute_identity(**other_hp)
            # this is hairy, but need to check if the dicts would be equal.
            if isinstance(self.hp, dict):
                # This should ideally never be the case, we would hope that
                # people are using HyperParameter objects in the Point tuples.
                hp_id = compute_identity(**self.hp)
            else:
                hp_id = self.hp.id()
            hps_equal = hp_id == other_id
        return hps_equal and self.perf == other[1]

    def __gt__(self, other: Tuple[object, ...]) -> bool:
        # Even though the tuple has (hp, perf), compare based on the order
        # (perf, hp).
        # This means that sorting a list of Points will work as expected!
        if isinstance(other, (Point, tuple)):
            hp, perf = other
            if not isinstance(perf, float):
                print(other)
                exit()
        return self.perf > perf

    # def __repr__(self):
    #     return super().__repr__()