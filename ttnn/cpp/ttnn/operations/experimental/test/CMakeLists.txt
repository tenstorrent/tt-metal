add_library(ttnn_op_experimental_test ${LIB_TYPE})
add_library(TT::NN::Ops::Experimental::Test ALIAS ttnn_op_experimental_test)

target_precompile_headers(ttnn_op_experimental_test REUSE_FROM TT::CommonPCH)

target_sources(ttnn_op_experimental_test PRIVATE hang_operation.cpp)

target_include_directories(ttnn_op_experimental_test PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_test
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_experimental_test LIBRARY COMPONENT tar)
