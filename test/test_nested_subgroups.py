from __future__ import annotations

# subgroups plan
from argparse import ArgumentParser, Namespace
from dataclasses import Field, dataclass, fields
from logging import getLogger as get_logger
from typing import Any, Sequence

from simple_parsing import subgroups

logger = get_logger(__name__)


@dataclass
class A:
    a: float = 0.0


@dataclass
class B:
    b: str = "bar"


@dataclass
class AB:
    a_or_b: A | B = subgroups({"a": A, "b": B}, default=A(1.23))


@dataclass
class C:
    c: bool = False


@dataclass
class D:
    d: int = 0


@dataclass
class CD:
    c_or_d: C | D = subgroups({"c": C, "d": D}, default=C(True))

    other_arg: str = "bob"


@dataclass
class Config:
    ab_or_cd: AB | CD = subgroups(
        {"ab": AB, "cd": CD},
        default=AB(a_or_b=B("heyo")),
    )


class ParserForSubgroups(ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.unresolved_subgroups: dict[str, Field] = {}
        # self.resolved_subgroups: dict[str, type] = {}
        self.dataclasses = {}

    def add_arguments(self, dataclass: type, dest: str):
        """Add arguments for a dataclass to the parser."""
        logger.info(f"add_arguments called for dataclass {dataclass} at dest {dest}")
        if dest in self.dataclasses:
            raise RuntimeError(
                f"Conflict: There's already a dataclass at dest {dest}: {self.dataclasses[dest]}"
            )

        dataclass_fields = fields(dataclass)
        subgroup_fields = [field for field in dataclass_fields if "subgroups" in field.metadata]

        self.dataclasses[dest] = dataclass

        for field in subgroup_fields:
            field_dest = f"{dest}.{field.name}"

            # Get the type to pass to `self.add_argument` from the dataclass field.
            field_type = field.type
            if "type" in field.metadata:
                field_type = field.metadata["type"]
            elif "custom_args" in field.metadata and "type" in field.metadata["custom_args"]:
                field_type = field.metadata["custom_args"]["type"]

            self.unresolved_subgroups[field_dest] = field
            choices = field.metadata["custom_args"]["choices"]

            # NOTE: ignoring name conflicts for now: the field dest is just going to be the field
            # name for now.
            option_string = f"--{field.name}"

            logger.info(
                f"Adding an argument for subgroup at dest {field_dest} with options {choices}"
            )
            self.add_argument(
                option_string,
                type=field_type,
                dest=field_dest,
                default=field.default,
                choices=choices,
            )

    def _add_rest_of_arguments(self):
        """Add arguments for all the non-subgroup fields."""
        for dest, dataclass_type in self.dataclasses.items():
            for field in fields(dataclass_type):
                field_dest = f"{dest}.{field.name}"

                if "subgroups" in field.metadata:
                    logger.debug(f"ignoring field at dest {field_dest} because it's a subgroup.")
                    continue
                field_type = field.type
                if isinstance(field.type, str):
                    # Need to evaluate the annotation.
                    from simple_parsing.annotation_utils.get_field_annotations import (
                        get_field_type_from_annotations,
                    )

                    field_type = get_field_type_from_annotations(dataclass_type, field.name)

                field_default = field.default
                add_argument_kwargs = dict(type=field_type, default=field_default)
                add_argument_kwargs.update(field.metadata.get("custom_args", {}))

                # NOTE: Option string is just the field name for now.
                option_string = f"--{field.name}"

                logger.info(f"add_argument('{option_string}', **{add_argument_kwargs})")
                self.add_argument(option_string, dest=field_dest, **add_argument_kwargs)

    def parse_known_args(self, args: Sequence | None = None, namespace: Namespace | None = None):
        for nesting_level in range(100):
            if not self.unresolved_subgroups:
                break

            # NOTE: Assuming that we are creating a new namespace for now. This might make a bit
            # simpler when doing this initial round of parsing, since we don't have to worry about
            # side-effects. Might need to test this later.
            assert namespace is None

            # Do rounds of parsing with just the subgroup arguments, until all the subgroups
            # are resolved to a dataclass type.

            # TODO: Could calling this repeatedly here have side-effects?

            parsed_args, unused_args = super().parse_known_args(args=args, namespace=namespace)
            logger.debug(
                f"Nesting level {nesting_level}: args: {args}, "
                f"parsed_args: {parsed_args}, unused_args: {unused_args}"
            )

            subgroups_at_this_level = self.unresolved_subgroups.copy()

            for dest, field in list(subgroups_at_this_level.items()):
                # NOTE: There should always be a parsed value for the subgroup argument on the
                # namespace. This is because we added all the subgroup arguments before we get
                # here.
                assert hasattr(parsed_args, dest)

                subgroup_chosen_value: str = getattr(parsed_args, dest)
                subgroup_chosen_class: type = field.metadata["subgroups"][subgroup_chosen_value]
                logger.info(
                    f"resolved the subgroup at dest {dest} to a value of "
                    f"{subgroup_chosen_value}, which means to use the "
                    f"{subgroup_chosen_class} dataclass."
                )
                self.unresolved_subgroups.pop(dest)
                self.add_arguments(dataclass=subgroup_chosen_class, dest=dest)

            if not self.unresolved_subgroups:
                logger.info("Done parsing all the subgroups!")
            else:
                logger.info(
                    f"Done parsing a round of subparsers, moving to the next nesting level, "
                    f"which appears to have {len(self.unresolved_subgroups)} unresolved subgroups."
                )

        logger.info("Adding the rest of the arguments.")
        self._add_rest_of_arguments()

        parsed_args, unused_args = super().parse_known_args(args=args, namespace=namespace)

        logger.info(f"Raw parsed args: {parsed_args}, unused args: {unused_args}")

        parsed_args_dict = vars(parsed_args)

        # --- Postprocessing step ---
        # Convert the dicts on the namespace to dataclasses.

        dests = [dest.split(".") for dest in self.dataclasses.keys()]
        assert all(dests)

        # Given the tree of constructor arguments, create the dataclasses, starting from the
        # most nested ones.

        # Sort the dict of dataclasses to process the most nested ones first.
        # We measure the "nesting level" by the number of "." in the dataclass dest.
        def nesting_level(dest: str) -> int:
            return len(dest.split("."))

        unresolved_dataclasses = dict(
            sorted(self.dataclasses.items(), key=lambda k_v: nesting_level(k_v[0]), reverse=True)
        )
        instantiated_dataclasses: dict[str, Any] = {}

        for dest, dataclass_type in unresolved_dataclasses.items():
            # Construct the dict of constructor arguments for this dataclass.
            # NOTE: Perhaps we could use the defaults from files here as an initial value, instead
            # of using the .set_defaults from argparse?
            dataclass_constructor_arguments = {}

            # Start by taking the parsed field values for "simple" fields from the namespace.
            for field in fields(dataclass_type):
                field_dest = f"{dest}.{field.name}"
                if field_dest in instantiated_dataclasses:
                    field_value = instantiated_dataclasses[field_dest]
                else:
                    field_value = parsed_args_dict[field_dest]

                logger.debug(
                    f"Field {field.name} of dataclass at dest {dest} has value {field_value}"
                )
                dataclass_constructor_arguments[field.name] = field_value

            logger.debug(
                f"Creating dataclass instance for dest {dest} with dataclass {dataclass_type} "
                f"and arguments: {dataclass_constructor_arguments}"
            )
            # Convert the dict to the dataclass.
            dataclass_instance = dataclass_type(**dataclass_constructor_arguments)

            instantiated_dataclasses[dest] = dataclass_instance

        # Set the instantiated dataclasses on the namespace.
        # However, we need to take care, and only actually need to only set the dataclasses that are not nested inside other
        # dataclasses!
        for dest, dataclass_type in instantiated_dataclasses.items():
            setattr(parsed_args, dest, dataclass_type)

        return parsed_args, unused_args


def test_subgroups():
    import shlex

    parser = ParserForSubgroups()
    parser.add_arguments(Config, "config")
    args = parser.parse_args(shlex.split("--ab_or_cd cd --c_or_d d --d 123"))
    assert args.config == Config(ab_or_cd=CD(c_or_d=D(d=123)))
