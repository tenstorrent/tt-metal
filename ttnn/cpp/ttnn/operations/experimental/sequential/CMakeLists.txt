# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

add_library(ttnn_op_experimental_sequential ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Sequential ALIAS ttnn_op_experimental_sequential)

target_precompile_headers(ttnn_op_experimental_sequential REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_sequential)

set_target_properties(
    ttnn_op_experimental_sequential
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_sequential
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
            sequential.hpp
            sequential_nanobind.hpp
            device/sequential_device_operation.hpp
            device/sequential_device_operation_types.hpp
            device/sequential_factory.hpp
            device/sequential_branch_descriptor.hpp
    PRIVATE
        sequential.cpp
        device/sequential_device_operation.cpp
        device/sequential_factory.cpp
)

target_include_directories(ttnn_op_experimental_sequential PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_sequential
    PUBLIC
        TTNN::Core
        TTNN::Ops::Experimental::Parallel
    PRIVATE
        TT::Metalium
)

install(TARGETS ttnn_op_experimental_sequential FILE_SET api COMPONENT ttnn-dev)
install(TARGETS ttnn_op_experimental_sequential LIBRARY COMPONENT tar)
