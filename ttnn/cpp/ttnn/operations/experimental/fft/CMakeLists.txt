# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

include(sources.cmake)

add_library(ttnn_op_experimental_fft ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::FFT ALIAS ttnn_op_experimental_fft)

tt_reuse_precompile_headers(ttnn_op_experimental_fft TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_fft)

set_target_properties(
    ttnn_op_experimental_fft
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Kernel sources for every backend reachable from
# ttnn::experimental::fft / ifft.  Layout follows the ttnn convention
# (matmul / sdpa / plusone / etc.):
#   device/kernels/dataflow/{*_reader,*_writer,*_common.h}
#   device/kernels/compute/*_compute.cpp
#
# Kernels are referenced from the ProgramSpec factories in
# device/*_factory.cpp via canonical paths
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
        device/single_tile_stockham_factory.cpp
        device/batched_stockham_factory.cpp
        device/apply_twiddles_device_operation.cpp
        device/apply_twiddles_factory.cpp
        device/apply_twiddles_xl_device_operation.cpp
        device/apply_twiddles_xl_factory.cpp
        device/transpose_rm_device_operation.cpp
        device/transpose_rm_factory.cpp
        device/rebank_rm_device_operation.cpp
        device/rebank_rm_factory.cpp
        device/rebank_rm_merge_device_operation.cpp
        device/rebank_rm_merge_factory.cpp
        device/fft_radix_pass_device_operation.cpp
        device/fft_radix_pass_factory.cpp
        device/complex_mul_device_operation.cpp
        device/complex_mul_factory.cpp
        fft.cpp
        apply_twiddles.cpp
        apply_twiddles_xl.cpp
        transpose_rm.cpp
        fft_radix_pass.cpp
        complex_mul.cpp
        bluestein.cpp
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

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_FFT_NANOBIND_SRCS})
endif()
