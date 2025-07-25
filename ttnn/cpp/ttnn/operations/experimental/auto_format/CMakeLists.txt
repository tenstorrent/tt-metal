add_library(ttnn_op_experimental_auto_format ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::AutoFormat ALIAS ttnn_op_experimental_auto_format)

target_precompile_headers(ttnn_op_experimental_auto_format REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_auto_format)

target_sources(
    ttnn_op_experimental_auto_format
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES auto_format.hpp
    PRIVATE
        auto_format.cpp
)

target_include_directories(ttnn_op_experimental_auto_format PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_auto_format PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_experimental_auto_format FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
