add_library(ttnn_op_experimental_copy ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Copy ALIAS ttnn_op_experimental_copy)

target_precompile_headers(ttnn_op_experimental_copy REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_copy)

target_sources(ttnn_op_experimental_copy PRIVATE typecast/typecast.cpp)

target_include_directories(ttnn_op_experimental_copy PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_copy
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_copy LIBRARY COMPONENT tar)
