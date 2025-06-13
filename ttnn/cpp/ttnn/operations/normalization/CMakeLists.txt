add_library(ttnn_op_normalization ${LIB_TYPE})
add_library(TTNN::Ops::Normalization ALIAS ttnn_op_normalization)

target_precompile_headers(ttnn_op_normalization REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_normalization)
set_target_properties(
    ttnn_op_normalization
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

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

file(GLOB_RECURSE kernels softmax/device/kernels/*)
target_sources(
    ttnn_op_normalization
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
)

target_include_directories(ttnn_op_normalization PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_normalization
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_normalization
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/normalization
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_normalization LIBRARY COMPONENT tar)
