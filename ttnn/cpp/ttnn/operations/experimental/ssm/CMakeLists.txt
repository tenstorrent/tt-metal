add_library(ttnn_op_experimental_ssm ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::SSM ALIAS ttnn_op_experimental_ssm)

target_precompile_headers(ttnn_op_experimental_ssm REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_ssm)

file(
    GLOB_RECURSE kernels
    hc_sum_reduce/device/kernels/*
    prefix_scan/device/kernels/*
    repeat_and_interleave_eltwise_mul/device/kernels/*
)

target_sources(
    ttnn_op_experimental_ssm
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        hc_sum_reduce/device/hc_sum_reduce_device_operation.cpp
        hc_sum_reduce/device/hc_sum_reduce_program_factory.cpp
        hc_sum_reduce/hc_sum_reduce.cpp
        prefix_scan/device/prefix_scan_device_operation.cpp
        prefix_scan/device/prefix_scan_program_factory.cpp
        prefix_scan/prefix_scan.cpp
        repeat_and_interleave_eltwise_mul/device/repeat_and_interleave_eltwise_mul_device_operation.cpp
        repeat_and_interleave_eltwise_mul/device/repeat_and_interleave_eltwise_mul_program_factory.cpp
        repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul.cpp
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
