add_library(ttnn_op_experimental_reduction ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Reduction ALIAS ttnn_op_experimental_reduction)

target_precompile_headers(ttnn_op_experimental_reduction REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_reduction)

target_sources(
    ttnn_op_experimental_reduction
    PRIVATE
        argmax/argmax.cpp
        fast_reduce_nc/device/fast_reduce_nc_device_operation.cpp
        fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp
        fast_reduce_nc/fast_reduce_nc.cpp
)

target_include_directories(ttnn_op_experimental_reduction PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_reduction
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_reduction LIBRARY COMPONENT tar)
