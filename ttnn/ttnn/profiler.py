# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
import os

import ttnn


def start_tracy_zone(source: str, functName: str, lineNum: int, color: int = 0):
    ttnn._ttnn.profiler.start_tracy_zone(source, functName, lineNum, color)


def stop_tracy_zone(name: str = "", color: int = 0):
    return ttnn._ttnn.profiler.stop_tracy_zone(name, color)


def tracy_message(source: str, color: int = 0xF0F8FF):
    ttnn._ttnn.profiler.tracy_message(source, color)


def tracy_frame():
    ttnn._ttnn.profiler.tracy_frame()


__all__ = []
