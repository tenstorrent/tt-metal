# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


def get_argument(position, name, args, kwargs):
    if position < len(args):
        output = args[position]
        args = args[:position] + args[position + 1 :]
    elif name in kwargs:
        output = kwargs[name]
        kwargs = {k: v for k, v in kwargs.items() if k != name}
    else:
        raise ValueError("Missing argument: {}".format(name))
    return output, args, kwargs
