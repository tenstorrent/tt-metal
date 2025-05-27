add_library(ttnn_op_experimental_copy ${LIB_TYPE})
add_library(TT::NN::Ops::Experimental::Copy ALIAS ttnn_op_experimental_copy)

target_precompile_headers(ttnn_op_experimental_copy REUSE_FROM TT::CommonPCH)

target_sources(ttnn_op_experimental_copy PRIVATE typecast/typecast.cpp)

target_include_directories(ttnn_op_experimental_copy PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_copy
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_experimental_copy LIBRARY COMPONENT tar)
