add_library(ttnn_op_reduction ${LIB_TYPE})
add_library(TTNN::Ops::Reduction ALIAS ttnn_op_reduction)

target_precompile_headers(ttnn_op_reduction REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_reduction)
set_target_properties(
    ttnn_op_reduction
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels argmax/device/kernels/*)
target_sources(
    ttnn_op_reduction
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        argmax/device/argmax_op.cpp
        argmax/device/argmax_program_factory.cpp
        argmax/argmax.cpp
        cumprod/cumprod.cpp
        cumprod/device/cumprod_device_operation.cpp
        cumprod/device/cumprod_program_factory.cpp
        cumsum/cumsum.cpp
        cumsum/device/cumsum_device_operation.cpp
        cumsum/device/cumsum_program_factory.cpp
        sampling/device/sampling_op.cpp
        sampling/device/sampling_program_factory.cpp
        sampling/sampling.cpp
        moe/device/moe_op.cpp
        moe/device/moe_program_factory.cpp
        moe/moe.cpp
        prod/device/prod_all_program_factory.cpp
        generic/device/single_core_hw/reduce_op_single_core_hw.cpp
        generic/device/multi_core_h/reduce_op_multi_core_h.cpp
        generic/device/multi_core_w/reduce_op_multi_core_w.cpp
        generic/device/reduce_op.cpp
        generic/generic_reductions.cpp
        prod/device/prod_nc_op.cpp
        prod/device/prod_nc_program_factory.cpp
        prod/device/prod_op_all.cpp
        prod/prod.cpp
        topk/device/topk_op.cpp
        topk/topk.cpp
        topk/device/topk_program_factory.cpp
        reduction_common/reduction_common.cpp
)

target_include_directories(ttnn_op_reduction PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_reduction
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_reduction
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/reduction
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_reduction LIBRARY COMPONENT tar)
