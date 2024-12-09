"""Simple, Elegant Argument parsing.

@author: Fabrice Normandin
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import itertools
import shlex
import sys
import typing
from argparse import SUPPRESS, Action, HelpFormatter, Namespace, _
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Sequence, Type, overload

from simple_parsing.helpers.subgroups import SubgroupKey
from simple_parsing.wrappers.dataclass_wrapper import DataclassWrapperType

from . import utils
from .conflicts import ConflictResolution, ConflictResolver
from .help_formatter import SimpleHelpFormatter
from .helpers.serialization.serializable import read_file
from .utils import (
    Dataclass,
    DataclassT,
    dict_union,
    is_dataclass_instance,
    is_dataclass_type,
)
from .wrappers import DashVariant, DataclassWrapper, FieldWrapper
from .wrappers.field_wrapper import ArgumentGenerationMode, NestedMode

logger = getLogger(__name__)


class ParsingError(RuntimeError, SystemExit):
    pass


class ArgumentParser(argparse.ArgumentParser):
    """Creates an ArgumentParser instance.

    Parameters
    ----------
    - conflict_resolution : ConflictResolution, optional

        What kind of prefixing mechanism to use when reusing dataclasses
        (argument groups).
        For more info, check the docstring of the `ConflictResolution` Enum.

    - add_option_string_dash_variants : DashVariant, optional

        Whether or not to add option_string variants where the underscores in
        attribute names are replaced with dashes.
        For example, when set to DashVariant.UNDERSCORE_AND_DASH,
        "--no-cache" and "--no_cache" can both be used to point to the same
        attribute `no_cache` on some dataclass.

    - argument_generation_mode : ArgumentGenerationMode, optional

        How to generate the arguments. In the ArgumentGenerationMode.FLAT mode,
        the default one, the arguments are flat when possible, ignoring
        their nested structure and including it only on the presence of a
        conflict.

        In the ArgumentGenerationMode.NESTED mode, the arguments are always
        composed reflecting their nested structure.

        In the ArgumentGenerationMode.BOTH mode, both kind of arguments
        are generated.

    - nested_mode : NestedMode, optional

        How to handle argument generation in for nested arguments
        in the modes ArgumentGenerationMode.NESTED and ArgumentGenerationMode.BOTH.
        In the NestedMode.DEFAULT mode, the nested arguments are generated
        reflecting their full 'destination' path from the returning namespace.

        In the NestedMode.WITHOUT_ROOT, the first level is removed. This is useful when
        parser.add_arguments is only called once, and where the same prefix would be shared
        by all arguments. For example, if you have a single dataclass MyArguments and
        you call parser.add_arguments(MyArguments, "args"), the arguments could look like this:
        '--args.input.path --args.output.path'.
        We could prefer to remove the root level in such a case
            so that the arguments get generated as
        '--input.path --output.path'.

    - formatter_class : Type[HelpFormatter], optional

        The formatter class to use. By default, uses
        `simple_parsing.SimpleHelpFormatter`, which is a combination of the
        `argparse.ArgumentDefaultsHelpFormatter`,
        `argparse.MetavarTypeHelpFormatter` and
        `argparse.RawDescriptionHelpFormatter` classes.

    - add_config_path_arg : bool, str, optional
        When set to `True`, adds a `--config_path` argument, of type Path, which is used to parse.
        If set to a string then this is the name of the config_path argument.

    - config_path: str, optional
        The values read from this file will overwrite the default values from the dataclass definitions.
        When `add_config_path_arg` is also set the defaults are first updated using `config_path`, and then
        updated with the contents of the `--config_path` file(s). By setting this value it will be default set
        `add_config_path_arg` to True.
    """

    def __init__(
        self,
        *args,
        parents: Sequence[ArgumentParser] = (),
        add_help: bool = True,
        conflict_resolution: ConflictResolution = ConflictResolution.AUTO,
        add_option_string_dash_variants: DashVariant = DashVariant.AUTO,
        argument_generation_mode=ArgumentGenerationMode.FLAT,
        nested_mode: NestedMode = NestedMode.DEFAULT,
        formatter_class: type[HelpFormatter] = SimpleHelpFormatter,
        add_config_path_arg: bool | str | None = None,
        config_path: Path | str | Sequence[Path | str] | None = None,
        add_dest_to_option_strings: bool | None = None,
        **kwargs,
    ):
        kwargs["formatter_class"] = formatter_class
        # Pass parents=[] since we override this mechanism below.
        # NOTE: We end up with the same parents.
        super().__init__(*args, parents=[], add_help=False, **kwargs)
        self.conflict_resolution = conflict_resolution

        # constructor arguments for the dataclass instances.
        # (a Dict[dest, [attribute, value]])
        # TODO: Stop using a defaultdict for the very important `self.constructor_arguments`!
        self.constructor_arguments: dict[str, dict[str, Any]] = defaultdict(dict)

        self._conflict_resolver = ConflictResolver(self.conflict_resolution)
        self._wrappers: list[DataclassWrapper] = []

        if add_dest_to_option_strings:
            argument_generation_mode = ArgumentGenerationMode.BOTH

        self._preprocessing_done: bool = False
        self.add_option_string_dash_variants = add_option_string_dash_variants
        self.argument_generation_mode = argument_generation_mode
        self.nested_mode = nested_mode

        FieldWrapper.add_dash_variants = add_option_string_dash_variants
        FieldWrapper.argument_generation_mode = argument_generation_mode
        FieldWrapper.nested_mode = nested_mode
        self._parents = tuple(parents)

        self.add_help = add_help
        if self.add_help:
            prefix_chars = self.prefix_chars
            default_prefix = "-" if "-" in prefix_chars else prefix_chars[0]
            self._help_action = super().add_argument(
                default_prefix + "h",
                default_prefix * 2 + "help",
                action="help",
                default=SUPPRESS,
                help=_("show this help message and exit"),
            )

        self.config_path = Path(config_path) if isinstance(config_path, str) else config_path
        if add_config_path_arg is None:
            # By default, add a config path argument if a config path was passed.
            add_config_path_arg = bool(config_path)
        self.add_config_path_arg = add_config_path_arg

    # TODO: Remove, since the base class already has nicer type hints.
    def add_argument(
        self,
        *name_or_flags: str,
        **kwargs,
    ) -> Action:
        return super().add_argument(
            *name_or_flags,
            **kwargs,
        )

    @overload
    def add_arguments(
        self,
        dataclass: type[DataclassT],
        dest: str,
        *,
        prefix: str = "",
        default: DataclassT | None = None,
        dataclass_wrapper_class: type[DataclassWrapper] = DataclassWrapper,
    ) -> DataclassWrapper[DataclassT]:
        pass

    @overload
    def add_arguments(
        self,
        dataclass: type[Dataclass],
        dest: str,
        *,
        prefix: str = "",
        dataclass_wrapper_class: type[DataclassWrapperType] = DataclassWrapper,
    ) -> DataclassWrapperType:
        pass

    @overload
    def add_arguments(
        self,
        dataclass: DataclassT,
        dest: str,
        *,
        prefix: str = "",
        default: None = None,
        dataclass_wrapper_class: type[DataclassWrapper] = DataclassWrapper,
    ) -> DataclassWrapper[DataclassT]:
        pass

    def add_arguments(
        self,
        dataclass: type[DataclassT] | DataclassT,
        dest: str,
        *,
        prefix: str = "",
        default: DataclassT | None = None,
        dataclass_wrapper_class: type[DataclassWrapperType] = DataclassWrapper,
    ) -> DataclassWrapper[DataclassT] | DataclassWrapperType:
        """Adds command-line arguments for the fields of `dataclass`.

        Parameters
        ----------
        dataclass : Union[Type[Dataclass], Dataclass]
            The dataclass whose fields are to be parsed from the command-line.
            If an instance of a dataclass is given, it is used as the default
            value if none is provided.
        dest : str
            The destination attribute of the `argparse.Namespace` where the
            dataclass instance will be stored after calling `parse_args()`
        prefix : str, optional
            An optional prefix to add prepend to the names of the argparse
            arguments which will be generated for this dataclass.
            This can be useful when registering multiple distinct instances of
            the same dataclass, by default ""
        default : Dataclass, optional
            An instance of the dataclass type to get default values from, by
            default None
        dataclass_wrapper_class : Type[DataclassWrapper], optional
            The type of `DataclassWrapper` to use for this dataclass. This can be used to customize
            how the arguments are generated. However, I'd suggest making a GitHub issue if you find
            yourself using this often.

        Returns
        -------
        The generated DataclassWrapper instance. Feel free to inspect / play around with this if
        you want :)
        """
        if is_dataclass_instance(dataclass):
            if default is not None:
                raise ValueError("Can't use `default` when `dataclass` is a dataclass instance.")
            dataclass = typing.cast(DataclassT, dataclass)
            dataclass_type = type(dataclass)
            default = dataclass
        else:
            if not is_dataclass_type(dataclass):
                raise ValueError(
                    f"`dataclass` should be a dataclass type or instance. Got {dataclass}."
                )
            dataclass = typing.cast(Type[DataclassT], dataclass)
            dataclass_type = dataclass
            default = default

        new_wrapper = self._add_arguments(
            dataclass_type=dataclass_type,
            name=dest,
            prefix=prefix,
            default=default,
            dataclass_wrapper_class=dataclass_wrapper_class,
        )
        self._wrappers.append(new_wrapper)
        return new_wrapper

    def parse_known_args(
        self,
        args: Sequence[str] | None = None,
        namespace: Namespace | None = None,
        attempt_to_reorder: bool = False,
    ):
        # NOTE: since the usual ArgumentParser.parse_args() calls
        # parse_known_args, we therefore just need to overload the
        # parse_known_args method to support both.
        if args is None:
            # args default to the system args
            args = sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)

        # default Namespace built from parser defaults
        if namespace is None:
            namespace = Namespace()
        if self.config_path:
            if isinstance(self.config_path, Path):
                config_paths = [self.config_path]
            else:
                config_paths = self.config_path
            for config_file in config_paths:
                self.set_defaults(config_file)

        if self.add_config_path_arg:
            config_path_name = (
                self.add_config_path_arg
                if isinstance(self.add_config_path_arg, str)
                else "config_path"
            )
            temp_parser = ArgumentParser(
                add_config_path_arg=False,
                add_help=False,
                add_option_string_dash_variants=FieldWrapper.add_dash_variants,
                argument_generation_mode=FieldWrapper.argument_generation_mode,
                nested_mode=FieldWrapper.nested_mode,
            )
            temp_parser.add_argument(
                f"--{config_path_name}",
                type=Path,
                nargs="*",
                default=self.config_path,
                help="Path to a config file containing default values to use.",
            )
            args_with_config_path, args = temp_parser.parse_known_args(args)
            config_path = getattr(args_with_config_path, config_path_name.replace("-", "_"))

            if config_path is not None:
                config_paths = config_path if isinstance(config_path, list) else [config_path]
                for config_file in config_paths:
                    self.set_defaults(config_file)

            # Adding it here just so it shows up in the help message. The default will be set in
            # the help string.
            self.add_argument(
                f"--{config_path_name}",
                type=Path,
                default=config_path,
                help="Path to a config file containing default values to use.",
            )

        assert isinstance(args, list)
        self._preprocessing(args=args, namespace=namespace)

        logger.debug(f"Parser {id(self)} is parsing args: {args}, namespace: {namespace}")
        parsed_args, unparsed_args = super().parse_known_args(args, namespace)

        if unparsed_args and self._subparsers and attempt_to_reorder:
            logger.warning(
                f"Unparsed arguments when using subparsers. Will "
                f"attempt to automatically re-order the unparsed arguments "
                f"{unparsed_args}."
            )
            index_in_start = args.index(unparsed_args[0])
            # Simply 'cycle' the args to the right ordering.
            new_start_args = args[index_in_start:] + args[:index_in_start]
            parsed_args, unparsed_args = super().parse_known_args(new_start_args)

        parsed_args = self._postprocessing(parsed_args)
        return parsed_args, unparsed_args

    def add_argument_group(
        self,
        title: str | None = None,
        description: str | None = None,
        prefix_chars=None,
        argument_default=None,
        conflict_handler=None,
    ) -> argparse._ArgumentGroup:
        return super().add_argument_group(
            title=title,
            description=description,
            prefix_chars=prefix_chars or self.prefix_chars,
            argument_default=argument_default or self.argument_default,
            conflict_handler=conflict_handler or self.conflict_handler,
        )

    def print_help(self, file=None, args: Sequence[str] | None = None):
        self._preprocessing(args=list(args) if args else [])
        return super().print_help(file)

    def set_defaults(self, config_path: str | Path | None = None, **kwargs: Any) -> None:
        """Set the default argument values, either from a config file, or from the given kwargs."""
        if config_path:
            defaults = read_file(config_path)
            if self.nested_mode == NestedMode.WITHOUT_ROOT and len(self._wrappers) == 1:
                # The file should have the same format as the command-line args, e.g. contain the
                # fields of the 'root' dataclass directly (e.g. "foo: 123"), rather a dict with
                # "config: foo: 123" where foo is a field of the root dataclass at dest 'config'.
                # Therefore, we add the prefix back here.
                defaults = {self._wrappers[0].dest: defaults}
                # We also assume that the kwargs are passed as foo=123
                kwargs = {self._wrappers[0].dest: kwargs}
            # Also include the values from **kwargs.
            kwargs = dict_union(defaults, kwargs)

        # The kwargs that are set in the dataclasses, rather than on the namespace.
        kwarg_defaults_set_in_dataclasses = {}
        for wrapper in self._wrappers:
            if wrapper.dest in kwargs:
                default_for_dataclass = kwargs[wrapper.dest]

                if isinstance(default_for_dataclass, (str, Path)):
                    default_for_dataclass = read_file(path=default_for_dataclass)
                elif not isinstance(default_for_dataclass, dict) and not dataclasses.is_dataclass(
                    default_for_dataclass
                ):
                    raise ValueError(
                        f"Got a default for field {wrapper.dest} that isn't a dataclass, dict or "
                        f"path: {default_for_dataclass}"
                    )

                # Set the .default attribute on the DataclassWrapper (which also updates the
                # defaults of the fields and any nested dataclass fields).
                wrapper.set_default(default_for_dataclass)

                # It's impossible for multiple wrappers in kwargs to have the same destination.
                assert wrapper.dest not in kwarg_defaults_set_in_dataclasses
                value_for_constructor_arguments = (
                    default_for_dataclass
                    if isinstance(default_for_dataclass, dict)
                    else dataclasses.asdict(default_for_dataclass)
                )
                kwarg_defaults_set_in_dataclasses[wrapper.dest] = value_for_constructor_arguments
                # Remove this from the **kwargs, so they don't get set on the namespace.
                kwargs.pop(wrapper.dest)
        # TODO: Stop using a defaultdict for the very important `self.constructor_arguments`!
        self.constructor_arguments = dict_union(
            self.constructor_arguments,
            kwarg_defaults_set_in_dataclasses,
            dict_factory=lambda: defaultdict(dict),
        )
        # For the rest of the values, use the default argparse behaviour (modifying the
        # self._defaults dictionary).
        super().set_defaults(**kwargs)

    def equivalent_argparse_code(self, args: Sequence[str] | None = None) -> str:
        """Returns the argparse code equivalent to that of `simple_parsing`.

        TODO: Could be fun, pretty sure this is useless though.

        Returns
        -------
        str
            A string containing the auto-generated argparse code.
        """
        self._preprocessing(list(args) if args else [])
        code = "parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)"
        for wrapper in self._wrappers:
            code += "\n"
            code += wrapper.equivalent_argparse_code()
            code += "\n"
        code += "args = parser.parse_args()\n"
        code += "print(args)\n"
        return code

    def _add_arguments(
        self,
        dataclass_type: type[DataclassT],
        name: str,
        *,
        prefix: str = "",
        dataclass_fn: Callable[..., DataclassT] | None = None,
        default: DataclassT | dict | None = None,
        dataclass_wrapper_class: type[DataclassWrapperType] = DataclassWrapper,
        parent: DataclassWrapper | None = None,
    ) -> DataclassWrapper[DataclassT] | DataclassWrapperType:
        assert is_dataclass_type(dataclass_type)
        assert (
            default is None
            or is_dataclass_instance(default)
            or default is argparse.SUPPRESS
            or isinstance(default, dict)
        )
        assert dataclass_fn is None or callable(dataclass_fn)

        for wrapper in self._wrappers:
            if wrapper.dest == name:
                if wrapper.dataclass == dataclass_type:
                    raise argparse.ArgumentError(
                        argument=None,
                        message=f"Destination attribute {name} is already used for "
                        f"dataclass of type {dataclass_type}. Make sure all destinations"
                        f" are unique. (new dataclass type: {dataclass_type})",
                    )
        if not isinstance(dataclass_type, type):
            if default is None:
                default = dataclass_type
            dataclass_type = type(dataclass_type)

        dataclass_fn = dataclass_fn or dataclass_type
        # Create this object that  holds the dataclass we will create arguments for and the
        # arguments that were passed.
        new_wrapper = dataclass_wrapper_class(
            dataclass=dataclass_type,
            name=name,
            prefix=prefix,
            default=default,
            parent=parent,
            dataclass_fn=dataclass_fn,
        )

        if new_wrapper.dest in self._defaults:
            new_wrapper.set_default(self._defaults[new_wrapper.dest])
        if self.nested_mode == NestedMode.WITHOUT_ROOT and all(
            field.name in self._defaults for field in new_wrapper.fields
        ):
            # If we did .set_defaults before we knew what dataclass we're using, then we try to
            # still make use of those defaults:
            new_wrapper.set_default(
                {
                    k: v
                    for k, v in self._defaults.items()
                    if k in [f.name for f in dataclasses.fields(new_wrapper.dataclass)]
                }
            )

        return new_wrapper

    def _preprocessing(self, args: Sequence[str] = (), namespace: Namespace | None = None) -> None:
        """Resolve potential conflicts, resolve subgroups, and add all the arguments."""
        logger.debug("\nPREPROCESSING\n")

        if self._preprocessing_done:
            return

        args = list(args)

        wrapped_dataclasses = self._wrappers.copy()
        # Fix the potential conflicts between dataclass fields with the same names.
        wrapped_dataclasses = self._conflict_resolver.resolve_and_flatten(wrapped_dataclasses)

        wrapped_dataclasses, chosen_subgroups = self._resolve_subgroups(
            wrappers=wrapped_dataclasses, args=args, namespace=namespace
        )

        # NOTE: We keep the subgroup fields in their dataclasses so they show up with the other
        # arguments.
        wrapped_dataclasses = _flatten_wrappers(wrapped_dataclasses)

        # Create one argument group per dataclass
        for wrapped_dataclass in wrapped_dataclasses:
            logger.debug(
                f"Parser {id(self)} is Adding arguments for dataclass: {wrapped_dataclass.dataclass} "
                f"at destinations {wrapped_dataclass.destinations}"
            )
            wrapped_dataclass.add_arguments(parser=self)

        self._wrappers = wrapped_dataclasses
        # Save this so we don't re-add all the arguments.
        self._preprocessing_done = True

    def _postprocessing(self, parsed_args: Namespace) -> Namespace:
        """Process the namespace by extract the fields and creating the objects.

        Instantiate the dataclasses from the parsed arguments and set them at
        their destination attribute in the namespace.

        Parameters
        ----------
        parsed_args : Namespace
            the result of calling `super().parse_args(...)` or
            `super().parse_known_args(...)`.
            TODO: Try and maybe return a nicer, typed version of parsed_args.


        Returns
        -------
        Namespace
            The original Namespace, with all the arguments corresponding to the
            dataclass fields removed, and with the added dataclass instances.
            Also keeps whatever arguments were added in the traditional fashion,
            i.e. with `parser.add_argument(...)`.
        """
        logger.debug("\nPOST PROCESSING\n")
        logger.debug(f"(raw) parsed args: {parsed_args}")

        self._remove_subgroups_from_namespace(parsed_args)
        # create the constructor arguments for each instance by consuming all
        # the relevant attributes from `parsed_args`
        wrappers = _flatten_wrappers(self._wrappers)

        constructor_arguments = self.constructor_arguments.copy()
        for wrapper in wrappers:
            for destination in wrapper.destinations:
                constructor_arguments.setdefault(destination, {})

        parsed_args, constructor_arguments = self._fill_constructor_arguments_with_fields(
            parsed_args, wrappers=wrappers, initial_constructor_arguments=constructor_arguments
        )
        parsed_args = self._instantiate_dataclasses(
            parsed_args, wrappers=wrappers, constructor_arguments=constructor_arguments
        )
        return parsed_args

    def _resolve_subgroups(
        self,
        wrappers: list[DataclassWrapper],
        args: list[str],
        namespace: Namespace | None = None,
    ) -> tuple[list[DataclassWrapper], dict[str, str]]:
        """Iteratively add and resolve all the choice of argument subgroups, if any.

        This modifies the wrappers in-place, by possibly adding children to the wrappers in the
        list.
        Returns a list with the modified wrappers.

        Each round does the following:
        1.  Resolve any conflicts using the conflict resolver. Two subgroups at the same nesting
            level, with the same name, get a different prefix, for example "--generator.optimizer"
            and "--discriminator.optimizer".
        2.  Add all the subgroup choice arguments to a parser.
        3.  Add the chosen dataclasses to the list of dataclasses to parse later in the main
            parser. This is done by adding wrapping the dataclass and adding it to the `wrappers`
            list.
        """

        unresolved_subgroups = _get_subgroup_fields(wrappers)
        # Dictionary of the subgroup choices that were resolved (key: subgroup dest, value: chosen
        # subgroup name).
        resolved_subgroups: dict[str, SubgroupKey] = {}

        if not unresolved_subgroups:
            # No subgroups to parse.
            return wrappers, {}

        # Use a temporary parser, to avoid parsing "vanilla argparse" arguments of `self` multiple
        # times.
        subgroup_choice_parser = argparse.ArgumentParser(
            add_help=False,
            # conflict_resolution=self.conflict_resolution,
            # add_option_string_dash_variants=self.add_option_string_dash_variants,
            # argument_generation_mode=self.argument_generation_mode,
            # nested_mode=self.nested_mode,
            formatter_class=self.formatter_class,
            # add_config_path_arg=self.add_config_path_arg,
            # config_path=self.config_path,
            # NOTE: We disallow abbreviations for subgroups for now. This prevents potential issues
            # for example if you have —a_or_b and A has a field —a then it will error out if you
            # pass —a=1 because 1 isn’t a choice for the a_or_b argument (because --a matches it
            # with the abbreviation feature turned on).
            allow_abbrev=False,
        )

        for current_nesting_level in itertools.count():
            # Do rounds of parsing with just the subgroup arguments, until all the subgroups
            # are resolved to a dataclass type.
            logger.debug(
                f"Starting subgroup parsing round {current_nesting_level}: {list(unresolved_subgroups.keys())}"
            )
            # Add all the unresolved subgroups arguments.
            for dest, subgroup_field in unresolved_subgroups.items():
                flags = subgroup_field.option_strings
                argument_options = subgroup_field.arg_options

                if subgroup_field.subgroup_default is dataclasses.MISSING:
                    assert argument_options["required"]
                else:
                    assert argument_options["default"] is subgroup_field.subgroup_default
                    assert not is_dataclass_instance(argument_options["default"])

                # TODO: Do we really need to care about this "SUPPRESS" stuff here?
                if argparse.SUPPRESS in subgroup_field.parent.defaults:
                    assert argument_options["default"] is argparse.SUPPRESS
                    argument_options["default"] = argparse.SUPPRESS

                logger.debug(
                    f"Adding subgroup argument: add_argument(*{flags} **{str(argument_options)})"
                )
                subgroup_choice_parser.add_argument(*flags, **argument_options)

            # Parse `args` repeatedly until all the subgroup choices are resolved.
            parsed_args, unused_args = subgroup_choice_parser.parse_known_args(
                args=args, namespace=namespace
            )
            logger.debug(
                f"Nesting level {current_nesting_level}: args: {args}, "
                f"parsed_args: {parsed_args}, unused_args: {unused_args}"
            )

            for dest, subgroup_field in list(unresolved_subgroups.items()):
                # NOTE: There should always be a parsed value for the subgroup argument on the
                # namespace. This is because we added all the subgroup arguments before we get
                # here.
                subgroup_dict = subgroup_field.subgroup_choices
                chosen_subgroup_key: SubgroupKey = getattr(parsed_args, dest)
                assert chosen_subgroup_key in subgroup_dict

                # Changing the default value of the (now parsed) field for the subgroup choice,
                # just so it shows (default: {chosen_subgroup_key}) on the command-line.
                # Note: This really isn't required, we could have it just be the default value, but
                # it seems a bit more consistent with us then showing the --help string for the
                # chosen dataclass type (as we're doing below).
                # subgroup_field.set_default(chosen_subgroup_key)
                logger.debug(
                    f"resolved the subgroup at {dest!r}: will use the subgroup at key "
                    f"{chosen_subgroup_key!r}"
                )

                default_or_dataclass_fn = subgroup_dict[chosen_subgroup_key]
                if is_dataclass_instance(default_or_dataclass_fn):
                    # The chosen value in the subgroup dict is a frozen dataclass instance.
                    default = default_or_dataclass_fn
                    dataclass_fn = functools.partial(dataclasses.replace, default)
                    dataclass_type = type(default)
                else:
                    default = None
                    dataclass_fn = default_or_dataclass_fn
                    dataclass_type = subgroup_field.field.metadata["subgroup_dataclass_types"][
                        chosen_subgroup_key
                    ]

                assert default is None or is_dataclass_instance(default)
                assert callable(dataclass_fn)
                assert is_dataclass_type(dataclass_type)

                name = dest.split(".")[-1]
                parent_dataclass_wrapper = subgroup_field.parent
                # NOTE: Using self._add_arguments so it returns the modified wrapper and doesn't
                # affect the `self._wrappers` list.
                new_wrapper = self._add_arguments(
                    dataclass_type=dataclass_type,
                    name=name,
                    dataclass_fn=dataclass_fn,
                    default=default,
                    parent=parent_dataclass_wrapper,
                )
                # Make the new wrapper a child of the class which contains the field.
                # - it isn't already a child
                # - it's parent is the parent dataclass wrapper
                # - the parent is already in the tree of DataclassWrappers.
                assert new_wrapper not in parent_dataclass_wrapper._children
                parent_dataclass_wrapper._children.append(new_wrapper)
                assert new_wrapper.parent is parent_dataclass_wrapper
                assert parent_dataclass_wrapper in _flatten_wrappers(wrappers)
                assert new_wrapper in _flatten_wrappers(wrappers)

                # Mark this subgroup as resolved.
                unresolved_subgroups.pop(dest)
                resolved_subgroups[dest] = chosen_subgroup_key
                # TODO: Should we remove the FieldWrapper for the subgroups now that it's been
                # resolved?

            # Find the new subgroup fields that weren't resolved before.
            # TODO: What if a name conflict occurs between a subgroup field and one of the new
            # fields below it? For example, something like --model model_a (and inside the `ModelA`
            # dataclass, there's a field called `model`. Then, this will cause a conflict!)
            # For now, I'm just going to wait and see how this plays out. I'm hoping that the
            # auto conflict resolution shouldn't run into any issues in this case.

            wrappers = self._conflict_resolver.resolve(wrappers)

            all_subgroup_fields = _get_subgroup_fields(wrappers)
            unresolved_subgroups = {
                k: v for k, v in all_subgroup_fields.items() if k not in resolved_subgroups
            }
            logger.debug(f"All subgroups: {list(all_subgroup_fields.keys())}")
            logger.debug(f"Resolved subgroups: {resolved_subgroups}")
            logger.debug(f"Unresolved subgroups: {list(unresolved_subgroups.keys())}")

            if not unresolved_subgroups:
                logger.debug("Done parsing all the subgroups!")
                break
            else:
                logger.debug(
                    f"Done parsing a round of subparsers at nesting level "
                    f"{current_nesting_level}. Moving to the next round which has "
                    f"{len(unresolved_subgroups)} unresolved subgroup choices."
                )
        return wrappers, resolved_subgroups

    def _remove_subgroups_from_namespace(self, parsed_args: argparse.Namespace) -> None:
        """Removes the subgroup choice results from the namespace.

        Modifies the namespace in-place.
        """
        # find all subgroup fields
        subgroup_fields = _get_subgroup_fields(self._wrappers)

        if not subgroup_fields:
            return
        # IDEA: Store the choices in a `subgroups` dict on the namespace.
        if not hasattr(parsed_args, "subgroups"):
            parsed_args.subgroups = {}

        for dest in subgroup_fields:
            chosen_value = getattr(parsed_args, dest)
            parsed_args.subgroups[dest] = chosen_value
            delattr(parsed_args, dest)

    def _instantiate_dataclasses(
        self,
        parsed_args: argparse.Namespace,
        wrappers: list[DataclassWrapper],
        constructor_arguments: dict[str, dict[str, Any]],
    ) -> argparse.Namespace:
        """Create the instances set them at their destination in the namespace.

        We now have all the constructor arguments for each instance.
        We can now sort out the dependencies, create the instances, and set them
        as attributes of the Namespace.

        Since the dataclasses might have nested children, and we need to pass
        all the constructor arguments when calling the dataclass constructors,
        we create the instances in a "bottom-up" fashion, creating the deepest
        objects first, and then setting their value in the
        `constructor_arguments` dict.

        Parameters
        ----------
        parsed_args : argparse.Namespace
            The 'raw' Namespace that is produced by `parse_args`.

        wrappers : list[DataclassWrapper]
            The (assumed flattened) list of dataclass wrappers that were created with
            `add_arguments`.

        constructor_arguments : dict[str, dict[str, Any]]
            The partially populated dict of constructor arguments for each dataclass. This will be
            consumed in order to create the dataclass instances for each DataclassWrapper.

        Returns
        -------
        argparse.Namespace
            The transformed namespace with the instances set at their
            corresponding destinations.
        """
        constructor_arguments = constructor_arguments.copy()
        # FIXME: There's a bug here happening with the `ALWAYS_MERGE` case: The namespace has the
        # values, but the constructor arguments dict doesn't.

        if self.conflict_resolution != ConflictResolution.ALWAYS_MERGE:
            assert len(wrappers) == len(constructor_arguments), "should have one dict per wrapper"

        # sort the wrappers so as to construct the leaf nodes first.
        sorted_dc_wrappers: list[DataclassWrapper] = sorted(
            wrappers, key=lambda w: w.nesting_level, reverse=True
        )
        assert len(sorted_dc_wrappers) == len(set(sorted_dc_wrappers))

        for dc_wrapper in sorted_dc_wrappers:
            logger.debug(f"Instantiating the wrapper with destinations {dc_wrapper.destinations}")

            for destination in dc_wrapper.destinations:
                logger.debug(f"Instantiating the dataclass at destination {destination}")
                # Instantiate the dataclass by passing the constructor arguments
                # to the constructor.
                constructor = dc_wrapper.dataclass_fn
                constructor_args = constructor_arguments.pop(destination)
                # If the dataclass wrapper is marked as 'optional' and all the
                # constructor args are None, then the instance is None.
                value_for_dataclass_field: Any | dict[str, Any] | None
                if argparse.SUPPRESS in dc_wrapper.defaults:
                    if constructor_args == {}:
                        value_for_dataclass_field = None
                    else:
                        # Don't create the dataclass instance. Instead, keep the value as a dict.
                        value_for_dataclass_field = constructor_args
                else:
                    value_for_dataclass_field = _create_dataclass_instance(
                        dc_wrapper, constructor, constructor_args
                    )

                if argparse.SUPPRESS in dc_wrapper.defaults and value_for_dataclass_field is None:
                    logger.debug(
                        f"Suppressing entire destination {destination} because none of its"
                        f"subattributes were specified on the command line."
                    )

                elif dc_wrapper.parent is not None:
                    parent_key, attr = utils.split_dest(destination)
                    logger.debug(
                        f"Setting a value of {value_for_dataclass_field} at attribute {attr} in "
                        f"parent at key {parent_key}."
                    )
                    constructor_arguments[parent_key][attr] = value_for_dataclass_field

                elif not hasattr(parsed_args, destination):
                    logger.debug(
                        f"setting attribute '{destination}' on the Namespace "
                        f"to a value of {value_for_dataclass_field}"
                    )
                    setattr(parsed_args, destination, value_for_dataclass_field)

                else:
                    # There is a collision: namespace already has an entry at this destination.
                    existing = getattr(parsed_args, destination)
                    if dc_wrapper.dest in self._defaults:
                        logger.debug(
                            f"Overwriting defaults in the namespace at destination '{destination}' "
                            f"on the Namespace ({existing}) to a value of {value_for_dataclass_field}"
                        )
                        setattr(parsed_args, destination, value_for_dataclass_field)
                    else:
                        raise RuntimeError(
                            f"Namespace should not already have a '{destination}' "
                            f"attribute!\n"
                            f"The value would be overwritten:\n"
                            f"- existing value: {existing}\n"
                            f"- new value:      {value_for_dataclass_field}"
                        )

        # We should be consuming all the constructor arguments.
        assert not constructor_arguments

        return parsed_args

    def _fill_constructor_arguments_with_fields(
        self,
        parsed_args: argparse.Namespace,
        wrappers: list[DataclassWrapper],
        initial_constructor_arguments: dict[str, dict[str, Any]],
    ) -> tuple[argparse.Namespace, dict[str, dict[str, Any]]]:
        """Create the constructor arguments for each instance.

        Creates the arguments by consuming all the attributes from
        `parsed_args`.
        Here we imitate a custom action, by having the FieldWrappers be
        callables that set their value in the `constructor_args` attribute.

        Parameters
        ----------
        parsed_args : argparse.Namespace
            the argparse.Namespace returned from super().parse_args().

        wrappers : list[DataclassWrapper]
            The (assumed flattened) list of dataclass wrappers that were created with
            `add_arguments`.

        constructor_arguments : dict[str, dict[str, Any]]
            The dict of constructor arguments to create for each dataclass. This will be filled by
            each FieldWrapper.

        Returns
        -------
        argparse.Namespace
            The namespace, without the consumed arguments.
        """

        if self.conflict_resolution != ConflictResolution.ALWAYS_MERGE:
            assert len(wrappers) == len(
                initial_constructor_arguments
            ), "should have one dict per wrapper"

        # The output
        constructor_arguments = initial_constructor_arguments.copy()

        parsed_arg_values = vars(parsed_args)
        deleted_values: dict[str, Any] = {}

        for wrapper in wrappers:
            for field in wrapper.fields:
                if argparse.SUPPRESS in wrapper.defaults and field.dest not in parsed_args:
                    continue

                if field.is_subgroup:
                    # Skip the subgroup fields, since we added a child DataclassWrapper for them.
                    logger.debug(f"Not calling the subgroup FieldWrapper for dest {field.dest}")
                    continue

                if not field.field.init:
                    # The field isn't an argument of the dataclass constructor.
                    continue

                # NOTE: If the field is reused (when using the ConflictResolution.ALWAYS_MERGE
                # strategy), then we store the multiple values in the `dest` of the first field.
                # They are they distributed in `constructor_arguments` using the
                # `field.destinations`, which gives the destination for each value.
                values = parsed_arg_values.pop(field.dest, field.default)
                deleted_values[field.dest] = values

                # call the "action" for the given attribute. This sets the right
                # value in the `constructor_arguments` dictionary.
                field(
                    parser=self,
                    namespace=parsed_args,
                    values=values,
                    constructor_arguments=constructor_arguments,
                )

        # "Clean up" the Namespace by returning a new Namespace without the
        # consumed attributes.
        leftover_args = argparse.Namespace(**parsed_arg_values)
        if deleted_values:
            logger.debug(f"deleted values: {deleted_values}")
            logger.debug(f"leftover args: {leftover_args}")

        return leftover_args, constructor_arguments

    @property
    def confilct_resolver_max_attempts(self) -> int:
        return self._conflict_resolver.max_attempts

    @confilct_resolver_max_attempts.setter
    def confilct_resolver_max_attempts(self, value: int):
        self._conflict_resolver.max_attempts = value


# TODO: Change the order of arguments to put `args` as the second argument.
def parse(
    config_class: type[DataclassT],
    config_path: Path | str | None = None,
    args: str | Sequence[str] | None = None,
    default: DataclassT | None = None,
    dest: str = "config",
    *,
    prefix: str = "",
    nested_mode: NestedMode = NestedMode.WITHOUT_ROOT,
    conflict_resolution: ConflictResolution = ConflictResolution.AUTO,
    add_option_string_dash_variants: DashVariant = DashVariant.AUTO,
    argument_generation_mode=ArgumentGenerationMode.FLAT,
    formatter_class: type[HelpFormatter] = SimpleHelpFormatter,
    add_config_path_arg: bool | str | None = None,
    **kwargs,
) -> DataclassT:
    """Parse the given dataclass from the command-line.

    See the `ArgumentParser` constructor for more details on the arguments (they are the same here
    except for `nested_mode`, which has a different default value).

    If `config_path` is passed, loads the values from that file and uses them as defaults.
    """
    if dest == add_config_path_arg:
        raise ValueError("`add_config_path_arg` cannot be the same as `dest`.")

    parser = ArgumentParser(
        nested_mode=nested_mode,
        add_help=True,
        config_path=config_path,
        conflict_resolution=conflict_resolution,
        add_option_string_dash_variants=add_option_string_dash_variants,
        argument_generation_mode=argument_generation_mode,
        formatter_class=formatter_class,
        add_config_path_arg=add_config_path_arg,
        **kwargs,
    )

    parser.add_arguments(config_class, prefix=prefix, dest=dest, default=default)

    if isinstance(args, str):
        args = shlex.split(args)
    parsed_args = parser.parse_args(args)

    config: Dataclass = getattr(parsed_args, dest)
    return config


def parse_known_args(
    config_class: type[Dataclass],
    config_path: Path | str | None = None,
    args: str | Sequence[str] | None = None,
    default: Dataclass | None = None,
    dest: str = "config",
    attempt_to_reorder: bool = False,
    *,
    nested_mode: NestedMode = NestedMode.WITHOUT_ROOT,
    conflict_resolution: ConflictResolution = ConflictResolution.AUTO,
    add_option_string_dash_variants: DashVariant = DashVariant.AUTO,
    argument_generation_mode=ArgumentGenerationMode.FLAT,
    formatter_class: type[HelpFormatter] = SimpleHelpFormatter,
    add_config_path_arg: bool | None = None,
) -> tuple[Dataclass, list[str]]:
    """Parse the given dataclass from the command-line, returning the leftover arguments.

    See the `ArgumentParser` constructor for more details on the arguments (they are the same here
    except for `nested_mode`, which has a different default value).

    If `config_path` is passed, loads the values from that file and uses them as defaults.
    """

    if isinstance(args, str):
        args = shlex.split(args)
    parser = ArgumentParser(
        nested_mode=nested_mode,
        add_help=True,
        # add_config_path_arg=None,
        config_path=config_path,
        conflict_resolution=conflict_resolution,
        add_option_string_dash_variants=add_option_string_dash_variants,
        argument_generation_mode=argument_generation_mode,
        formatter_class=formatter_class,
        add_config_path_arg=add_config_path_arg,
    )
    parser.add_arguments(config_class, dest=dest, default=default)
    parsed_args, unknown_args = parser.parse_known_args(
        args, attempt_to_reorder=attempt_to_reorder
    )
    config: Dataclass = getattr(parsed_args, dest)
    return config, unknown_args


def _get_subgroup_fields(wrappers: list[DataclassWrapper]) -> dict[str, FieldWrapper]:
    subgroup_fields = {}
    all_wrappers = _flatten_wrappers(wrappers)
    for wrapper in all_wrappers:
        for field in wrapper.fields:
            if field.is_subgroup:
                assert field not in subgroup_fields.values()
                subgroup_fields[field.dest] = field
    return subgroup_fields


def _remove_duplicates(wrappers: list[DataclassWrapper]) -> list[DataclassWrapper]:
    return list(set(wrappers))


def _assert_no_duplicates(wrappers: list[DataclassWrapper]) -> None:
    if len(wrappers) != len(set(wrappers)):
        raise RuntimeError(
            "Duplicate wrappers found! This is a potentially nasty bug on our "
            "part. Please make an issue at https://www.github.com/lebrice/SimpleParsing/issues "
        )


def _flatten_wrappers(wrappers: list[DataclassWrapper]) -> list[DataclassWrapper]:
    """Takes a list of nodes, returns a flattened list of all nodes in the tree."""
    _assert_no_duplicates(wrappers)
    roots_only = _unflatten_wrappers(wrappers)
    return sum(([w] + list(w.descendants) for w in roots_only), [])


def _unflatten_wrappers(wrappers: list[DataclassWrapper]) -> list[DataclassWrapper]:
    """Given a list of nodes in one or more trees, returns only the root nodes.

    In our context, this is all the dataclass arg groups that were added with
    `parser.add_arguments`.
    """
    _assert_no_duplicates(wrappers)
    return [w for w in wrappers if w.parent is None]


def _create_dataclass_instance(
    wrapper: DataclassWrapper[DataclassT],
    constructor: Callable[..., DataclassT],
    constructor_args: dict[str, Any],
) -> DataclassT | None:
    # Check if the dataclass annotation is marked as Optional.
    # In this case, if no arguments were passed, and the default value is None, then return
    # None.
    # TODO: (BUG!) This doesn't distinguish the case where the defaults are passed via the
    # command-line from the case where no arguments are passed at all!
    if wrapper.optional and wrapper.default is None:
        for field_wrapper in wrapper.fields:
            arg_value = constructor_args[field_wrapper.name]
            default_value = field_wrapper.default
            logger.debug(
                f"field {field_wrapper.name}, arg value: {arg_value}, "
                f"default value: {default_value}"
            )
            if arg_value != default_value:
                # Value is not the default value, so an argument must have been passed.
                # Break, and return the instance.
                break
        else:
            logger.debug(f"All fields for {wrapper.dest} were either at their default, or None.")
            return None
    logger.debug(f"Calling constructor: {constructor}(**{constructor_args})")
    return constructor(**constructor_args)
