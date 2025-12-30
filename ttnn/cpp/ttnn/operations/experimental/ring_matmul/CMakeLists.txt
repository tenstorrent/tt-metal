# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

add_library(ttnn_op_experimental_ring_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::RingMatmul ALIAS ttnn_op_experimental_ring_matmul)

target_precompile_headers(ttnn_op_experimental_ring_matmul REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_ring_matmul)

target_sources(
    ttnn_op_experimental_ring_matmul
    PRIVATE
        ring_matmul.cpp
        device/ring_matmul_device_operation.cpp
        device/ring_matmul_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_ring_matmul PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_ring_matmul
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_ring_matmul LIBRARY COMPONENT tar)
