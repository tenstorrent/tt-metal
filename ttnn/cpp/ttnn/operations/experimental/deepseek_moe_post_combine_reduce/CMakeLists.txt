# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

add_library(ttnn_op_experimental_deepseek_moe_post_combine_reduce ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepseekMoEPostCombineReduce
    ALIAS ttnn_op_experimental_deepseek_moe_post_combine_reduce
)

target_precompile_headers(ttnn_op_experimental_deepseek_moe_post_combine_reduce REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_moe_post_combine_reduce)

set_target_properties(
    ttnn_op_experimental_deepseek_moe_post_combine_reduce
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_moe_post_combine_reduce
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/deepseek_moe_post_combine_reduce_device_operation.cpp
        device/deepseek_moe_post_combine_reduce_program_factory.cpp
        deepseek_moe_post_combine_reduce.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_moe_post_combine_reduce PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_moe_post_combine_reduce
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_moe_post_combine_reduce
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_reduce
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_deepseek_moe_post_combine_reduce LIBRARY COMPONENT tar)
