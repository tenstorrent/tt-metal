add_library(ttnn_op_experimental_copy ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Copy ALIAS ttnn_op_experimental_copy)

if(TT_METAL_BUILD_AGGRESSIVE_PCH)
    target_precompile_headers(ttnn_op_experimental_copy REUSE_FROM ttnn_core)
else()
    target_precompile_headers(ttnn_op_experimental_copy REUSE_FROM TT::CommonPCH)
endif()
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
