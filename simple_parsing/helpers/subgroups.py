from __future__ import annotations

import functools
import inspect
import types
from dataclasses import _MISSING_TYPE, MISSING
from enum import Enum
from logging import getLogger as get_logger
from typing import Callable, TypeVar, overload

from simple_parsing.utils import Dataclass, DataclassT, is_dataclass_type

logger = get_logger(__name__)

# TODO: Change this to a bound of Hashable.
# It seems to consider `default`
Key = TypeVar("Key", str, int, bool, Enum)
OtherDataclassT = TypeVar("OtherDataclassT", bound=Dataclass)


@overload
def subgroups(
    subgroups: dict[Key, Callable[..., DataclassT]],
    *args,
    default: Key,
    default_factory: _MISSING_TYPE = MISSING,
    **kwargs,
) -> DataclassT:
    ...


# TODO: Enable this overload if we make `subgroups` more flexible (see below).
# @overload
# def subgroups(
#     subgroups: Mapping[Key, type[DataclassT]],
#     *args,
#     default_factory: Callable[[], OtherDataclassT],
#     **kwargs,
# ) -> DataclassT | OtherDataclassT:
#     ...


@overload
def subgroups(
    subgroups: dict[Key, Callable[..., DataclassT]],
    *args,
    default: _MISSING_TYPE = MISSING,
    default_factory: Callable[[], DataclassT],
    **kwargs,
) -> DataclassT:
    ...


@overload
def subgroups(
    subgroups: dict[Key, Callable[..., DataclassT]],
    *args,
    default: _MISSING_TYPE = MISSING,
    default_factory: _MISSING_TYPE = MISSING,
    **kwargs,
) -> DataclassT:
    ...


def subgroups(
    subgroups: dict[Key, Callable[..., DataclassT]],
    *args,
    default: Key | _MISSING_TYPE = MISSING,
    default_factory: Callable[[], DataclassT] | _MISSING_TYPE = MISSING,
    **kwargs,
) -> DataclassT:
    """Creates a field that will be a choice between different subgroups of arguments.

    This is different than adding a subparser action. There can only be one subparser action, while
    there can be arbitrarily many subgroups. Subgroups can also be nested!

    TODO: We don't yet inspect the default values for the fields of each subgroup, when the
    subgroup values aren't classes (e.g. `functools.partial`s). The help text is therefore slightly
    wrong: it shows the field defaults from the class definition, not those of the subgroup.

    Parameters
    ----------
    subgroups :
        Dictionary mapping from the subgroup name to the subgroup type.
    default :
        The default subgroup to use, by default MISSING, in which case a subgroup has to be
        selected. Needs to be a key in the subgroups dictionary.
    default_factory :
        The default_factory to use to create the subgroup. Needs to be a value of the `subgroups`
        dictionary.

    Returns
    -------
    A field whose type is the Union of the different possible subgroups.
    """
    if default_factory is not MISSING and default is not MISSING:
        raise ValueError("Can't pass both default and default_factory!")
    if default is not MISSING and default not in subgroups:
        raise ValueError("default must be a key in the subgroups dict!")

    metadata = kwargs.pop("metadata", {})
    metadata["subgroups"] = subgroups
    metadata["subgroup_default"] = default

    for value in subgroups.values():
        if isinstance(value, types.LambdaType):
            raise NotImplementedError(
                "Lambda expressions can't currently be used as subgroup values, since we're "
                "unable to inspect which dataclass they return without invoking them.\n"
                "If you want to choose between different versions of a dataclass where arguments "
                "change between subgroups, consider using a `functools.partial` instead. "
            )

    if default_factory is not MISSING and default_factory not in list(subgroups.values()):
        # NOTE: This is because we need to have a "default key" to associate with the
        # default_factory (and we set that as the default value for the argument of this field).
        raise ValueError("`default_factory` must be a value in the subgroups dict.")
    # IDEA: We could add a `default` key for this `default_factory` value into the `subgroups`
    # dict? However if it's a lambda expression, then we wouldn't then be able to inspect the
    # return type of that default factory (see above). Therefore there doesn't seem to be any
    # good way to allow lambda expressions as default factories yet. Perhaps I'm
    # overcomplicating things and it's actually very simple to do. I'll have to think about it.

    choices = subgroups.keys()

    # NOTE: Perhaps we could raise a warning if the default_factory is a Lambda, since we have to
    # instantiate that value in order to inspect the attributes and its values..

    # try:
    #     subgroup_dataclasses = {
    #         key: _get_dataclass_type_from_callable(value) for key, value in subgroups.items()
    #     }
    #     subgroup_dataclass_fns = subgroups.copy()
    #     # IDEA: These here would be used to update the defaults in the help string.
    #     subgroup_dataclass_defaults = {}
    # except Exception as exc:
    #     raise ValueError(
    #         "We were unable to figure out the dataclasses to use for each subgroup!\n"
    #         "Please make an issue on GitHub, and include the following traceback:\n" + str(exc)
    #     ) from exc
    # default_factory_dataclass = None
    # if default_factory is not MISSING:
    #     default_factory_dataclass = _get_dataclass_type_from_callable(default_factory)
    # subgroup_field_values = {}

    if default is not MISSING:
        assert default in subgroups.keys()
        default_factory = subgroups[default]
        metadata["subgroup_default"] = default
        default = MISSING

    elif default_factory is not MISSING:
        # assert default_factory in subgroups.values()
        # default_factory passed, which is in the subgroups dict. Find the matching key.
        matching_keys = [k for k, v in subgroups.items() if v is default_factory]
        if not matching_keys:
            # Use == instead of `is` this time.
            matching_keys = [k for k, v in subgroups.items() if v == default_factory]

        # We wouldn't get here if default_factory wasn't in the subgroups dict values.
        assert matching_keys
        if len(matching_keys) > 1:
            raise ValueError(
                f"Default subgroup {default} is found more than once in the subgroups dict?"
            )
        subgroup_default = matching_keys[0]
        metadata["subgroup_default"] = subgroup_default
    else:
        # Store `MISSING` as the subgroup default.
        metadata["subgroup_default"] = MISSING

    from .fields import choice

    return choice(choices, *args, default=default, default_factory=default_factory, metadata=metadata, **kwargs)  # type: ignore


def _get_dataclass_type_from_callable(dataclass_fn: Callable[..., DataclassT]) -> type[DataclassT]:
    """Inspects and returns the type of dataclass that the given callable will return."""
    if is_dataclass_type(dataclass_fn):
        return dataclass_fn

    if isinstance(dataclass_fn, functools.partial):
        if is_dataclass_type(dataclass_fn.func):
            return dataclass_fn.func
        # partial to a function that should return a dataclass. Hopefully it has a return type
        # annotation, otherwise we'd have to call the function just to know the return type!
        signature = inspect.signature(dataclass_fn)

        if signature.return_annotation is inspect.Signature.empty:
            raise TypeError(
                f"Unable to determine what type of dataclass would be returned by the callable "
                f"{dataclass_fn!r}, because it doesn't have a return type annotation."
            )
        # NOTE: Could recurse here, so it also works with `partial(partial(...))` and
        # `partial(some_function)`
        # Recurse, so this also works with partial(partial(...)) (idk why you'd do that though.)

        if isinstance(signature.return_annotation, str):
            raise NotImplementedError(
                f"TODO: Evaluate the annotation {signature.return_annotation!r} into a type and "
                f"figure out the dataclass that corresponds to it using the function's namespace."
            )
        return _get_dataclass_type_from_callable(dataclass_fn=dataclass_fn.func)
    # Lambda expression
    if isinstance(dataclass_fn, types.LambdaType):
        signature = inspect.signature(dataclass_fn)
        # TODO: How can we check what the dataclass type is?
        assert False, (dataclass_fn, signature.return_annotation, inspect.getsource(dataclass_fn))
