# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set(TTNN_OP_EXPERIMENTAL_PAGED_POOL_GATHER_API_HEADERS
    paged_pool_gather.hpp
    device/paged_pool_gather_device_operation.hpp
    device/paged_pool_gather_device_operation_types.hpp
    device/paged_pool_gather_program_factory.hpp
)

set(TTNN_OP_EXPERIMENTAL_PAGED_POOL_GATHER_SRCS
    device/paged_pool_gather_device_operation.cpp
    device/paged_pool_gather_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_PAGED_POOL_GATHER_NANOBIND_SRCS paged_pool_gather_nanobind.cpp)
