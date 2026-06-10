# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set(TTNN_OP_EXPERIMENTAL_TOPK_XL_API_HEADERS
    topk_xl.hpp
    device/topk_xl_device_operation.hpp
    device/topk_xl_device_operation_types.hpp
    device/topk_xl_program_factory.hpp
)

set(TTNN_OP_EXPERIMENTAL_TOPK_XL_SRCS
    device/topk_xl_device_operation.cpp
    device/topk_xl_program_factory.cpp
)
