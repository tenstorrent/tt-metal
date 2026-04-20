include(sources.cmake)

add_library(ttnn_op_experimental_reduction ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Reduction ALIAS ttnn_op_experimental_reduction)

tt_reuse_precompile_headers(ttnn_op_experimental_reduction TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_reduction)

set_target_properties(
    ttnn_op_experimental_reduction
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    deepseek_grouped_gate/device/kernels/*
    deepseek_moe_fast_reduce_nc/device/kernels/*
    fast_reduce_nc/device/kernels/*
    integral_image/device/kernels/*
)

target_sources(
    ttnn_op_experimental_reduction
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
        ${TTNN_OP_EXPERIMENTAL_REDUCTION_SRCS}
)

target_include_directories(ttnn_op_experimental_reduction PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_reduction
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_reduction
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/reduction
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_reduction LIBRARY COMPONENT tar)
