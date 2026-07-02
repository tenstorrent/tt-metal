# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

add_library(ttnn_op_experimental_deepseek_moe_generalized_moe_gate ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeek::MoE::GeneralizedMoeGate
    ALIAS ttnn_op_experimental_deepseek_moe_generalized_moe_gate
)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_moe_generalized_moe_gate TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_moe_generalized_moe_gate)

target_sources(
    ttnn_op_experimental_deepseek_moe_generalized_moe_gate
    PRIVATE
        generalized_moe_gate.cpp
        device/generalized_moe_gate_device_operation.cpp
        device/generalized_moe_gate_program_descriptor_builder.cpp
        device/generalized_moe_gate_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_moe_generalized_moe_gate PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_moe_generalized_moe_gate
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_deepseek_moe_generalized_moe_gate LIBRARY COMPONENT tar)
