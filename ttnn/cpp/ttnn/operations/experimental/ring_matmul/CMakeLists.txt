# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set(RING_MATMUL_LIB_SRCS device/ring_matmul_program_factory.cpp)

add_library(ring_matmul_lib ${RING_MATMUL_LIB_SRCS})

target_include_directories(
    ring_matmul_lib
    PUBLIC
        ${TTNN_INCLUDES}
        ${PROJECT_SOURCE_DIR}/ttnn/cpp/ttnn/operations/experimental/ring_matmul/device
)

target_link_libraries(
    ring_matmul_lib
    PUBLIC
        tt_metal
        ttnn
        matmul_lib
)
