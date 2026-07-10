# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set(TTNN_OP_EXPERIMENTAL_TOPK_LARGE_INDICES_API_HEADERS
    topk_large_indices.hpp
    device/topk_large_indices_device_operation.hpp
    device/topk_large_indices_device_operation_types.hpp
    device/topk_large_indices_program_factory.hpp
)

set(TTNN_OP_EXPERIMENTAL_TOPK_LARGE_INDICES_SRCS
    device/topk_large_indices_device_operation.cpp
    device/topk_large_indices_program_factory.cpp
)
