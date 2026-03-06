# SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Auto-optimal configuration selection for TTNN operations."""


def matmul_auto(*args, **kwargs):
    from ttnn.operations.auto_config.matmul_auto import matmul_auto as _fn
    return _fn(*args, **kwargs)


def __getattr__(name):
    if name == "AutoConfigSelector":
        from ttnn.operations.auto_config.base import AutoConfigSelector
        return AutoConfigSelector
    if name == "ConfigCache":
        from ttnn.operations.auto_config.config_cache import ConfigCache
        return ConfigCache
    raise AttributeError(name)


__all__ = ["matmul_auto", "AutoConfigSelector", "ConfigCache"]
