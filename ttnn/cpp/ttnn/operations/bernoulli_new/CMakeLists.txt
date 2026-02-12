add_library(ttnn_op_bernoulli_new ${LIB_TYPE})
add_library(TTNN::Ops::BernoulliNew ALIAS ttnn_op_bernoulli_new)

target_precompile_headers(ttnn_op_bernoulli_new REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_bernoulli_new)

set_target_properties(
    ttnn_op_bernoulli_new
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_bernoulli_new
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
    PRIVATE
        bernoulli_new.cpp
        device/bernoulli_new_device_operation.cpp
        device/bernoulli_new_program_factory.cpp
)

target_include_directories(ttnn_op_bernoulli_new PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_bernoulli_new
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_bernoulli_new LIBRARY COMPONENT tar)
