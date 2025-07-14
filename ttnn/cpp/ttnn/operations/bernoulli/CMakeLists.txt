add_library(ttnn_op_bernoulli ${LIB_TYPE})
add_library(TTNN::Ops::Bernoulli ALIAS ttnn_op_bernoulli)

target_precompile_headers(ttnn_op_bernoulli REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_bernoulli)

target_sources(
    ttnn_op_bernoulli
    PRIVATE
        bernoulli.cpp
        device/bernoulli_device_operation.cpp
        device/bernoulli_program_factory.cpp
)

target_include_directories(ttnn_op_bernoulli PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_bernoulli
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_bernoulli LIBRARY COMPONENT tar)
