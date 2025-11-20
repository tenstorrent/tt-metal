#
# command_console.py
#
# Interactive console that accepts keyboard input from the user.
#

from __future__ import annotations
from dataclasses import dataclass, field
import itertools
from typing import Any, Callable, Dict, List, Tuple, Type


@dataclass
class Param:
    name: str  # parameter name
    type: Type = None  # type of the parameter (int, float, bool, str only)
    values: List[str] = None  # if not None, a list of allowed values
    range: Tuple[Any, Any] = None  # if type == int or float, range of acceptable values
    default: Any = None  # if not None, this is an optional param with a default value


@dataclass
class Command:
    handler: Callable[[Dict[str, Any] | None], None]  # handler that takes an optional dictionary of parameters
    command: str
    params: List[CommandConsole.Param] = field(default_factory=list)
    description: str | None = None


class CommandConsole:
    def __init__(self, commands: List[Command]):
        self._running = True

        # Validate that built-in commands are not re-defined
        built_in_commands = ["q", "quit", "exit", "?", "help"]
        redefined_commands = set()
        for definition in commands:
            if definition.command in built_in_commands:
                redefined_commands.add(definition.command)
        if len(redefined_commands) > 0:
            redefined_commands_msg = ", ".join([f"'{command}'" for command in redefined_commands])
            plural = len(redefined_commands) > 1
            raise ValueError(
                f"{redefined_commands_msg} {'are' if plural else 'is a'} built-in command{'s' if plural else ''} and may not be redefined"
            )

        # Validate params have been defined correctly
        for description in commands:
            # May not redefine help!
            if description.command in ["q", "quit", "exit", "?", "help"]:
                raise ValueError(f"'{description.command}' is a built-in command and may not be redefined")

            # Optional params may not appear before non-optional ones
            optional_encountered = False
            names = set()
            for param in description.params:
                if optional_encountered and param.default is None:
                    raise ValueError(
                        f"Command '{description.command}' is ill-defined: optional parameters must follow non-optional ones"
                    )
                optional_encountered = optional_encountered or (param.default is not None)
                if param.name in names:
                    raise ValueError(f"Command '{description.command}' has multiple parameters named '{param.name}'")
                names.add(param.name)
                if param.type not in [str, int, float, bool]:
                    raise ValueError(
                        f"Command '{description.command}' has invalid type for parameter '{param.name}'. Must be one of: str, int, float, or bool."
                    )

        # Inject built-in commands
        self._commands: List[Command] = [
            Command(handler=lambda params: self._quit(), command="q"),
            Command(handler=lambda params: self._quit(), command="quit"),
            Command(handler=lambda params: self._quit(), command="exit", description="Quit"),
            Command(handler=lambda params: self._help(), command="?"),
            Command(handler=lambda params: self._help(), command="help", description="Print help"),
        ] + commands

        # Build dictionary
        self._command_by_name: Dict[str, Command] = {description.command: description for description in self._commands}

    def run(self):
        print("Type 'quit' to exit and 'help' for a list of commands.")
        while self._running:
            # Read command and parse it
            words = self._get_line_as_words()
            if len(words) == 0:
                continue
            command, arg_strings = words[0], words[1:]
            result = self._parse_command(command=command, args=arg_strings)
            if result is None:
                continue
            command_definition, args = result

            # Invoke command handler
            command_definition.handler(args)

    def _quit(self):
        self._running = False

    def _help(self):
        print("Commands:")
        print("---------")
        for command_definition in self._commands:
            params = command_definition.params
            print(f"{command_definition.command} ", end="")
            param_names = [self._param_description(param=param) for param in params]
            print(" ".join(param_names))
            if command_definition.description is not None:
                print(f"  {command_definition.description}")
        print("")

    @staticmethod
    def _param_description(param: Param) -> str:
        return "".join(
            [
                "<" if param.default is None else "[",
                param.name,
                ":",
                param.type.__name__,
                ("=" + "|".join(param.values)) if param.values is not None else "",
                ">" if param.default is None else "]",
            ]
        )

    def _get_line_as_words(self, prompt: str = ">>") -> List[str]:
        print(prompt, end="")
        line = input().rstrip("\n").strip()
        return line.split()

    def _parse_command(self, command: str, args: List[str]) -> Tuple[Command, Dict[str, Any]] | None:
        parsed_args: Dict[str, Any] = {}
        definition = self._command_by_name.get(command)
        if definition is None:
            print("Invalid command. Use 'help' for a list of commands.")
            return None
        for i, param in enumerate(definition.params):
            out_of_bounds = i >= len(args)
            is_required = param.default is None
            if out_of_bounds:
                if is_required:
                    print(f"Error: Missing required parameter: {param.name}")
                    return None
                else:
                    parsed_args[param.name] = param.default
            else:
                value = self._try_parse_value(arg=args[i], param=param)
                if value is None:
                    return None
                parsed_args[param.name] = value
        return definition, parsed_args

    @staticmethod
    def _try_parse_value(arg: str, param: Param) -> Any | None:
        if param.values is not None and len(param.values) > 0:
            # Arg must be one of the specified values
            if arg not in param.values:
                print(f"Error: Parameter '{param.name}' must be one of: {', '.join(param.values)}")
                return None
        if param.type == int or param.type == float:
            value = 0
            try:
                value = int(arg) if param.type == int else float(arg)
            except ValueError:
                required_type = "an integer" if param.type == int else "a float"
                print(f"Error: Parameter '{param.name}' must be {required_type} value")
                return None
            if param.range is not None:
                if value < min(param.range) or value > max(param.range):
                    print(f"Error: Parameter '{param.name}' must be in range: [{min(param.range)},{max(param.range)}]")
                    return None
            return value
        elif param.type == bool:
            value = False
            try:
                if arg.lower() in ["on", "enable", "enabled", "true", "t"]:
                    value = True
                elif arg.lower() in ["off", "disable", "disabled", "false", "f"]:
                    value = False
                else:
                    value = int(arg) != 0
            except ValueError:
                print(f"Error: Parameter '{param.name}' must be boolean value")
                return None
            return value
        else:
            # Assume str
            return arg

    @staticmethod
    def _serialize_values_array(values: List[float]) -> str:
        if len(values) == 1:
            return str(values[0])
        elif len(values) <= 5:
            return "[ " + ", ".join([str(value) for value in values]) + " ]"
        else:
            return f"[ {values[0]}, {values[1]}, ..., {values[-2]}, {values[-1]} ] ({len(values)} samples)"
