# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

set(DEEPSEEK_MOE_POST_COMBINE_REDUCE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/deepseek_moe_post_combine_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deepseek_moe_post_combine_reduce_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/deepseek_moe_post_combine_reduce_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/deepseek_moe_post_combine_reduce_program_factory.cpp
)

add_library(deepseek_moe_post_combine_reduce_lib OBJECT ${DEEPSEEK_MOE_POST_COMBINE_REDUCE_SOURCES})
target_link_libraries(deepseek_moe_post_combine_reduce_lib PUBLIC ttnn_lib)
set_property(TARGET deepseek_moe_post_combine_reduce_lib PROPERTY POSITION_INDEPENDENT_CODE ON)