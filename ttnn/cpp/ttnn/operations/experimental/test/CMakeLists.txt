add_library(ttnn_op_experimental_test ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Test ALIAS ttnn_op_experimental_test)

if(TT_METAL_BUILD_AGGRESSIVE_PCH)
    target_precompile_headers(ttnn_op_experimental_test REUSE_FROM ttnn_core)
else()
    target_precompile_headers(ttnn_op_experimental_test REUSE_FROM TT::CommonPCH)
endif()

target_sources(ttnn_op_experimental_test PRIVATE hang_device/hang_device_operation.cpp)

target_include_directories(ttnn_op_experimental_test PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_test
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_test LIBRARY COMPONENT tar)
