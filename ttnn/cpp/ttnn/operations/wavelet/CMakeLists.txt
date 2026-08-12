include(sources.cmake)

add_library(ttnn_op_wavelet ${LIB_TYPE})
add_library(TTNN::Ops::Wavelet ALIAS ttnn_op_wavelet)

tt_reuse_precompile_headers(ttnn_op_wavelet TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_wavelet)

set_target_properties(
    ttnn_op_wavelet
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    common/*.hpp
    device/kernels/*.cpp
    device/kernels/*.h
    device/kernels/*.hpp
    device/protocol/*.hpp
    generated/schemes/*.hpp
    planner/static_scheme.hpp
    planner/step.hpp
)

target_sources(
    ttnn_op_wavelet
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_WAVELET_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_WAVELET_SRCS}
)

# The implementation sources instantiate all 106 static schemes. Compile them
# outside the unity source to avoid anonymous-namespace collisions and an
# excessively large translation unit.
set_source_files_properties(
    device/wavelet_1d_operation_impl.cpp
    device/wavelet_2d_operation_impl.cpp
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            ON
)

target_include_directories(ttnn_op_wavelet PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_wavelet PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_wavelet
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/wavelet
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_wavelet LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_WAVELET_NANOBIND_SRCS})
endif()
