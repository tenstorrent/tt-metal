include(sources.cmake)

add_library(ttnn_op_experimental_ssm ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::SSM ALIAS ttnn_op_experimental_ssm)

tt_reuse_precompile_headers(ttnn_op_experimental_ssm TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_ssm)

set_target_properties(
    ttnn_op_experimental_ssm
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    hc_sum_reduce/device/kernels/*
    prefix_scan/device/kernels/*
    repeat_and_interleave_eltwise_mul/device/kernels/*
)

target_sources(
    ttnn_op_experimental_ssm
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
        ${TTNN_OP_EXPERIMENTAL_SSM_SRCS}
)

target_include_directories(ttnn_op_experimental_ssm PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_ssm
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_ssm
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/ssm
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_ssm LIBRARY COMPONENT tar)
