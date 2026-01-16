add_library(ttnn_op_reduction ${LIB_TYPE})
add_library(TTNN::Ops::Reduction ALIAS ttnn_op_reduction)

target_precompile_headers(ttnn_op_reduction REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_reduction)
set_target_properties(
    ttnn_op_reduction
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    accumulation/device/kernels/*
    argmax/device/kernels/*
    generic/device/kernels/*
    moe/device/kernels/*
    prod/device/kernels/*
    sampling/device/kernels/*
    topk/device/kernels/*
    manual_seed/device/kernels/*
)
target_sources(
    ttnn_op_reduction
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES argmax/argmax.hpp generic/generic_reductions.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        argmax/device/argmax_device_operation.cpp
        argmax/device/argmax_multi_core_program_factory.cpp
        argmax/device/argmax_single_core_program_factory.cpp
        argmax/argmax.cpp
        accumulation/accumulation_common.cpp
        accumulation/cumprod/cumprod.cpp
        accumulation/cumsum/cumsum.cpp
        accumulation/ema/ema.cpp
        accumulation/ema/device/ema_device_operation.cpp
        accumulation/ema/device/ema_program_factory.cpp
        accumulation/device/accumulation_device_operation.cpp
        accumulation/device/accumulation_program_factory.cpp
        sampling/device/sampling_device_operation.cpp
        sampling/device/sampling_program_factory.cpp
        sampling/sampling.cpp
        moe/device/moe_device_operation.cpp
        moe/device/moe_program_factory.cpp
        moe/moe.cpp
        prod/device/prod_all_program_factory.cpp
        generic/device/reduce_op_device_operation.cpp
        generic/device/reduce_op_single_core_hw_program_factory.cpp
        generic/device/reduce_op_multi_core_h_program_factory.cpp
        generic/device/reduce_op_multi_core_w_program_factory.cpp
        generic/device/reduce_op.cpp
        generic/device/common.cpp
        generic/generic_reductions.cpp
        prod/device/prod_all_device_operation.cpp
        prod/device/prod_nc_op.cpp
        prod/device/prod_nc_device_operation.cpp
        prod/device/prod_nc_program_factory.cpp
        prod/device/prod_op_all.cpp
        prod/prod.cpp
        topk/device/topk_device_operation.cpp
        topk/device/topk_utils.cpp
        topk/topk.cpp
        topk/device/topk_single_core_program_factory.cpp
        topk/device/topk_multi_core_program_factory.cpp
        reduction_common/reduction_common.cpp
        manual_seed/manual_seed.cpp
        manual_seed/device/manual_seed_operation.cpp
        manual_seed/device/manual_seed_program_factory.cpp
)

target_include_directories(ttnn_op_reduction PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_reduction PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_reduction
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/reduction
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_reduction LIBRARY COMPONENT tar)
