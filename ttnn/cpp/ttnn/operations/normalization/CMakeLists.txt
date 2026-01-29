add_library(ttnn_op_normalization ${LIB_TYPE})
add_library(TTNN::Ops::Normalization ALIAS ttnn_op_normalization)

target_precompile_headers(ttnn_op_normalization REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_normalization)
set_target_properties(
    ttnn_op_normalization
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    batch_norm/device/kernels/*
    groupnorm/device/kernels/*
    softmax/device/kernels/*
    kernel_util/compute/*
    kernel_util/dataflow/*
    kernel_util/generic/*
)
target_sources(
    ttnn_op_normalization
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
            softmax/device/softmax_operation_types.hpp
            softmax/device/softmax_device_operation.hpp
            softmax/device/softmax_program_factory_general.hpp
            softmax/device/softmax_program_factory_general_w_small.hpp
            softmax/device/softmax_program_factory_general_w_large.hpp
            softmax/device/softmax_program_factory_general_h_small.hpp
            softmax/device/softmax_program_factory_general_h_large.hpp
            softmax/device/softmax_program_factory_general_c_large.hpp
            softmax/device/softmax_program_factory_attention_optimized.hpp
            softmax/device/softmax_program_factory_attention_optimized_sharded.hpp
            softmax/softmax.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        batch_norm/batch_norm.cpp
        batch_norm/device/batch_norm_device_operation.cpp
        batch_norm/device/batch_norm_program_factory.cpp
        batch_norm/device/running_statistics_device_operation.cpp
        batch_norm/device/running_statistics_program_factory.cpp
        batch_norm/device/batch_norm_utils.cpp
        groupnorm/device/groupnorm_program_utils.cpp
        groupnorm/device/groupnorm_device_operation.cpp
        groupnorm/device/groupnorm_sharded_program_factory.cpp
        groupnorm/device/groupnorm_no_mcast_program_factory.cpp
        groupnorm/device/groupnorm_mcast_program_factory.cpp
        groupnorm/groupnorm.cpp
        groupnorm/groupnorm_input_mask.cpp
        softmax/softmax.cpp
        softmax/device/softmax_device_operation.cpp
        softmax/device/softmax_program_factory_general.cpp
        softmax/device/softmax_program_factory_general_w_small.cpp
        softmax/device/softmax_program_factory_general_w_large.cpp
        softmax/device/softmax_program_factory_general_h_small.cpp
        softmax/device/softmax_program_factory_general_h_large.cpp
        softmax/device/softmax_program_factory_general_c_large.cpp
        softmax/device/softmax_program_factory_attention_optimized.cpp
        softmax/device/softmax_program_factory_attention_optimized_sharded.cpp
)

target_include_directories(ttnn_op_normalization PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_normalization PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_normalization
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/normalization
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_normalization LIBRARY COMPONENT tar)
