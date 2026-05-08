add_library(ttnn_op_experimental_dropout_new ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DropoutNew ALIAS ttnn_op_experimental_dropout_new)

tt_reuse_precompile_headers(ttnn_op_experimental_dropout_new TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_dropout_new)

set_target_properties(
    ttnn_op_experimental_dropout_new
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_dropout_new
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES dropout_new.hpp
    PRIVATE
        device/dropout_new_device_operation.cpp
        device/dropout_new_program_factory.cpp
        dropout_new.cpp
)

target_include_directories(ttnn_op_experimental_dropout_new PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_dropout_new PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_experimental_dropout_new FILE_SET api COMPONENT ttnn-dev)

install(TARGETS ttnn_op_experimental_dropout_new LIBRARY COMPONENT tar)
