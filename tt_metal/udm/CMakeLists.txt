# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set(UDM_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_tensor_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_circular_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_semaphore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_utils.cpp
)

add_library(tt_metal_udm STATIC ${UDM_SRCS})

target_include_directories(tt_metal_udm PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(
    tt_metal_udm
    PUBLIC
        metal_common_libs
        tt_metal
)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include/tt_metal FILES_MATCHING PATTERN "*.hpp")
