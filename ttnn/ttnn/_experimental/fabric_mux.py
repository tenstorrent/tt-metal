# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Program-descriptor configuration for TT-Metal's experimental fabric mux."""

import ttnn

_fabric_mux = ttnn._ttnn.operations.experimental.fabric_mux

ChannelType = _fabric_mux.ChannelType
Config = _fabric_mux.Config
KernelBuildOptLevel = _fabric_mux.KernelBuildOptLevel
channel_buffer_size_bytes = _fabric_mux.channel_buffer_size_bytes
client_compile_time_args = _fabric_mux.client_compile_time_args
client_runtime_args = _fabric_mux.client_runtime_args

__all__ = [
    "ChannelType",
    "Config",
    "KernelBuildOptLevel",
    "channel_buffer_size_bytes",
    "client_compile_time_args",
    "client_runtime_args",
]
