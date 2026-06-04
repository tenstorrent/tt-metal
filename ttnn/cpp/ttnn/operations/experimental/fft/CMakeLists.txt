# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

add_library(ttnn_op_experimental_fft ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::FFT ALIAS ttnn_op_experimental_fft)

target_precompile_headers(ttnn_op_experimental_fft REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_fft)

set_target_properties(
    ttnn_op_experimental_fft
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Kernel sources for every backend reachable from
# ttnn::experimental::fft / ifft (stockham, universal_xl, universal,
# universal_bf16). Layout follows the ttnn convention (matmul / sdpa /
# plusone / etc.):
#   device/kernels/dataflow/{*_reader,*_writer,*_common.h}
#   device/kernels/compute/*_compute.cpp
#
# These kernels are referenced directly by the in-tree orchestrator
# headers (device/{stockham,universal,universal_xl,universal_bf16}_host.hpp)
# via canonical paths
# `ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/...`.
# See device/kernels/README.md for the full file → backend mapping.
file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_fft
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
        device/fft_device_operation.cpp
        device/fft_program_factory.cpp
        fft.cpp
)

target_include_directories(ttnn_op_experimental_fft PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_fft
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_fft
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/fft
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_fft LIBRARY COMPONENT tar)
