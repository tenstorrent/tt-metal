add_library(ttnn_op_experimental_dropout ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Dropout ALIAS ttnn_op_experimental_dropout)

target_precompile_headers(ttnn_op_experimental_dropout REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_dropout)

target_sources(
    ttnn_op_experimental_dropout
    PRIVATE
        device/dropout_device_operation.cpp
        device/dropout_program_factory.cpp
        dropout.cpp
)

target_include_directories(ttnn_op_experimental_dropout PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_dropout
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_dropout LIBRARY COMPONENT tar)
