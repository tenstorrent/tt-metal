add_library(ttnn_op_experimental_reduction ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Reduction ALIAS ttnn_op_experimental_reduction)

target_precompile_headers(ttnn_op_experimental_reduction REUSE_FROM TTNN::PCH)
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
        fast_reduce_nc/device/fast_reduce_nc_device_operation.cpp
        fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp
        fast_reduce_nc/fast_reduce_nc.cpp
        deepseek_moe_fast_reduce_nc/device/deepseek_moe_fast_reduce_nc_device_operation.cpp
        deepseek_moe_fast_reduce_nc/device/deepseek_moe_fast_reduce_nc_program_factory.cpp
        deepseek_moe_fast_reduce_nc/deepseek_moe_fast_reduce_nc.cpp
        integral_image/device/intimg_device_operation.cpp
        integral_image/device/intimg_program_factory.cpp
        integral_image/intimg.cpp
        deepseek_grouped_gate/deepseek_grouped_gate.cpp
        deepseek_grouped_gate/device/deepseek_grouped_gate_device_operation.cpp
        deepseek_grouped_gate/device/deepseek_grouped_gate_program_factory.cpp
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
