include(sources.cmake)

add_library(ttnn_op_experimental_nvfp4 ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Nvfp4 ALIAS ttnn_op_experimental_nvfp4)

tt_reuse_precompile_headers(ttnn_op_experimental_nvfp4 TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_nvfp4)

set_target_properties(
    ttnn_op_experimental_nvfp4
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)

target_sources(
    ttnn_op_experimental_nvfp4
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_NVFP4_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_NVFP4_SRCS}
)

target_link_libraries(ttnn_op_experimental_nvfp4 PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_nvfp4
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/dequantize_nvfp4
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_nvfp4 FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Register bindings on the shared Python module when Python bindings are enabled.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_NVFP4_NANOBIND_SRCS})
endif()
