# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
import os

import ttnn
from contextlib import contextmanager
import inspect


def start_tracy_zone(source: str, functName: str, lineNum: int, color: int = 0):
    ttnn._ttnn.profiler.start_tracy_zone(source, functName, lineNum, color)


def stop_tracy_zone(name: str = "", color: int = 0):
    return ttnn._ttnn.profiler.stop_tracy_zone(name, color)


def tracy_message(source: str, color: int = 0xF0F8FF):
    ttnn._ttnn.profiler.tracy_message(source, color)


def tracy_frame():
    ttnn._ttnn.profiler.tracy_frame()


@contextmanager
def tracy_zone(description: str):
    frame = inspect.currentframe().f_back.f_back  # Go up two frames
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno

    if description is None:
        function_name = frame.f_code.co_name
        description = f"{function_name}"

    try:
        ttnn.start_tracy_zone(filename, description, lineno)
        # print(f"tracy start zone {filename}, {description}, {lineno}")
        yield
    finally:
        ttnn.stop_tracy_zone()


__all__ = []
