add_library(ttnn_op_uniform ${LIB_TYPE})
add_library(TTNN::Ops::Uniform ALIAS ttnn_op_uniform)

target_precompile_headers(ttnn_op_uniform REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_uniform)

target_sources(
    ttnn_op_uniform
    PRIVATE
        device/uniform_device_operation.cpp
        device/uniform_program_factory.cpp
        uniform.cpp
)

target_include_directories(ttnn_op_uniform PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_uniform
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_uniform LIBRARY COMPONENT tar)
