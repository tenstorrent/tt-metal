add_library(ttnn_op_index_fill ${LIB_TYPE})
add_library(TT::NN::Ops::IndexFill ALIAS ttnn_op_index_fill)

target_precompile_headers(ttnn_op_index_fill REUSE_FROM TT::CommonPCH)

target_sources(
    ttnn_op_index_fill
    PRIVATE
        device/index_fill_device_operation.cpp
        device/index_fill_multi_core_factory.cpp
        index_fill.cpp
)

target_include_directories(ttnn_op_index_fill PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_index_fill
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_index_fill LIBRARY COMPONENT tar)
