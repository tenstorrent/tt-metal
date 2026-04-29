# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

add_library(ttnn_op_experimental_gpt_oss_swiglu ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::GptOssSwiglu ALIAS ttnn_op_experimental_gpt_oss_swiglu)

target_precompile_headers(ttnn_op_experimental_gpt_oss_swiglu REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_gpt_oss_swiglu)

target_sources(
    ttnn_op_experimental_gpt_oss_swiglu
    PRIVATE
        device/gpt_oss_swiglu_device_operation.cpp
        device/gpt_oss_swiglu_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_gpt_oss_swiglu PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_gpt_oss_swiglu
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_gpt_oss_swiglu LIBRARY COMPONENT tar)
