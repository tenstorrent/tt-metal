# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

add_library(ttnn_op_experimental_prefill_moe_compute ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::PrefillMoeCompute ALIAS ttnn_op_experimental_prefill_moe_compute)

target_precompile_headers(ttnn_op_experimental_prefill_moe_compute REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_prefill_moe_compute)

target_sources(
    ttnn_op_experimental_prefill_moe_compute
    PRIVATE
        device/prefill_moe_compute_device_operation.cpp
        device/prefill_moe_compute_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_prefill_moe_compute PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_prefill_moe_compute
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_prefill_moe_compute LIBRARY COMPONENT tar)
