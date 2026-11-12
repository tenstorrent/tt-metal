# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Experimental and subject to change. TTNN command-list parameters: op arguments a built TTNN command list can update.

import ttnn
from ttnn._ttnn.operations.command_list import is_recording, pop_cmd_list_binds, push_cmd_list_binds


class CmdListParam:
    """Marks an op argument as a TTNN command-list parameter (experimental).

    Pass it by keyword to an op recorded inside ``with builder:``. The op runs with ``default`` and the recorded
    argument becomes updatable with ``ttnn.experimental.CommandList.update_args({param: value})`` or
    ``{name: value}``. Outside a recording the op simply runs with ``default``.

    Args:
        default (ttnn.Tensor | float | int): Value the op is called and recorded with.
        name (str): Key for ``ttnn.experimental.CommandList.update_args``. Ops passed the same name share one
            parameter.
    """

    __slots__ = ("default", "name")
    # Lets Operation.__call__ skip the argument scan in programs that never create one.
    any_created = False

    def __init__(self, default, name):
        if not isinstance(name, str) or not name:
            raise ValueError("CmdListParam name must be a non-empty string")
        if isinstance(default, bool) or not isinstance(default, (ttnn.Tensor, float, int)):
            raise TypeError("CmdListParam default must be a ttnn.Tensor, float, or int")
        self.default = default
        self.name = name
        CmdListParam.any_created = True

    def __repr__(self):
        return f"CmdListParam(name={self.name!r})"


def has_cmd_list_param(args, kwargs):
    for value in args:
        if type(value) is CmdListParam:
            return True
    for value in kwargs.values():
        if type(value) is CmdListParam:
            return True
    return False


def call_with_cmd_list_params(operation, args, kwargs):
    # Only a direct argument is a parameter, not one inside a list or other container.
    for value in args:
        if type(value) is CmdListParam:
            raise TypeError(
                f"{operation.python_fully_qualified_name}: CmdListParam {value.name!r} must be passed by keyword"
            )
    binds = []
    unwrapped_kwargs = {}
    for key, value in kwargs.items():
        if type(value) is CmdListParam:
            binds.append((value.name, key, value.default))
            value = value.default
        unwrapped_kwargs[key] = value

    if not is_recording():
        return operation(*args, **unwrapped_kwargs)

    push_cmd_list_binds(operation.python_fully_qualified_name, binds)
    try:
        output = operation(*args, **unwrapped_kwargs)
    except BaseException:
        pop_cmd_list_binds(validate=False)
        raise
    pop_cmd_list_binds(validate=True)
    return output
