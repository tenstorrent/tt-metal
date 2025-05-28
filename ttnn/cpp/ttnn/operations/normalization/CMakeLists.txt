add_library(ttnn_op_normalization ${LIB_TYPE})
add_library(TT::NN::Ops::Normalization ALIAS ttnn_op_normalization)

target_precompile_headers(ttnn_op_normalization REUSE_FROM TT::CommonPCH)

target_sources(
    ttnn_op_normalization
    PRIVATE
        batch_norm/batch_norm.cpp
        batch_norm/device/batch_norm_device_operation.cpp
        batch_norm/device/batch_norm_program_factory.cpp
        batch_norm/device/running_statistics_device_operation.cpp
        batch_norm/device/running_statistics_program_factory.cpp
        groupnorm/device/groupnorm_op.cpp
        groupnorm/device/multi_core/groupnorm_op_multi_core.cpp
        groupnorm/groupnorm.cpp
        layernorm/device/layernorm_op.cpp
        layernorm/device/multi_core/layernorm_op_multi_core.cpp
        layernorm/layernorm.cpp
        layernorm_distributed/device/layernorm_post_all_gather_op.cpp
        layernorm_distributed/device/layernorm_pre_all_gather_op.cpp
        layernorm_distributed/device/multi_core/layernorm_post_all_gather_op_multi_core.cpp
        layernorm_distributed/device/multi_core/layernorm_pre_all_gather_op_multi_core.cpp
        layernorm_distributed/layernorm_post_all_gather.cpp
        layernorm_distributed/layernorm_pre_all_gather.cpp
        rmsnorm/rmsnorm.cpp
)

target_include_directories(ttnn_op_normalization PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_normalization
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_normalization LIBRARY COMPONENT tar)
