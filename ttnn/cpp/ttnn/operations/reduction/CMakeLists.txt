add_library(ttnn_op_reduction ${LIB_TYPE})
add_library(TT::NN::Ops::Reduction ALIAS ttnn_op_reduction)

target_precompile_headers(ttnn_op_reduction REUSE_FROM TT::CommonPCH)

target_sources(
    ttnn_op_reduction
    PRIVATE
        argmax/device/argmax_op.cpp
        argmax/device/argmax_program_factory.cpp
        argmax/argmax.cpp
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
        TT::NN::Core
)

install(TARGETS ttnn_op_reduction LIBRARY COMPONENT tar)
