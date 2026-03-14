add_library(ttnn_op_normal ${LIB_TYPE})
add_library(TTNN::Ops::Normal ALIAS ttnn_op_normal)

target_precompile_headers(ttnn_op_normal REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_normal)

# Verify only the api FILE_SET
set_target_properties(
    ttnn_op_normal
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_normal
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES normal.hpp
    PRIVATE
        normal.cpp
)

target_include_directories(ttnn_op_normal PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_normal PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_normal FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
