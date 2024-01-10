# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


def make_class_member_fn(TensorClass, symbols, function_names, nargs):
    functions = list(filter(lambda x: x, [v for k, v in symbols.items() if k in function_names]))
    for fn in functions:
        handle = (
            lambda self, *_: ((len(*_) + 1) >= nargs)
            and fn(self, *_)
            or ValueError(f"Too few arguments for {TensorClass.__name__}.{fn.__name__}")
        )
        setattr(TensorClass, fn.__name__.split(".")[-1], handle)
    return
