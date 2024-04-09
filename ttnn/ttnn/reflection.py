# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


def pop_argument(name, args, kwargs):
    if args:
        output, *args = args
    elif name in kwargs:
        output = kwargs[name]
        kwargs = {k: v for k, v in kwargs.items() if k != name}
    else:
        raise ValueError("Missing argument: {}".format(name))
    return output, args, kwargs
