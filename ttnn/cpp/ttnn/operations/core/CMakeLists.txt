add_library(ttnn_op_core ${LIB_TYPE})
add_library(TTNN::Ops::Core ALIAS ttnn_op_core)

target_precompile_headers(ttnn_op_core REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_core)

target_sources(
    ttnn_op_core
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
            core.hpp
            to_dtype/to_dtype_op.hpp
            to_layout/to_layout_op.hpp
            compute_kernel/compute_kernel_config.hpp
            to_memory_config/to_memory_config_op.hpp
            set_tensor_spec/set_tensor_spec.hpp
    PRIVATE
        compute_kernel/compute_kernel_config.cpp
        core.cpp
        to_layout/to_layout_op.cpp
        to_dtype/to_dtype_op.cpp
        set_tensor_spec/set_tensor_spec.cpp
)

target_include_directories(ttnn_op_core PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_core
    PUBLIC
        TTNN::Core
        TT::Metalium
)

install(TARGETS ttnn_op_core FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
