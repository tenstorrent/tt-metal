add_library(ttnn_op_creation ${LIB_TYPE})
add_library(TTNN::Ops::Creation ALIAS ttnn_op_creation)

target_precompile_headers(ttnn_op_creation REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_creation)

set_target_properties(
    ttnn_op_creation
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_creation
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
            creation.hpp
    PRIVATE
)

target_include_directories(ttnn_op_creation PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_creation
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_creation LIBRARY COMPONENT tar)
