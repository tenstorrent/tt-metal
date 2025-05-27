add_library(ttnn_op_full ${LIB_TYPE})
add_library(TT::NN::Ops::Full ALIAS ttnn_op_full)

target_precompile_headers(ttnn_op_full REUSE_FROM TT::CommonPCH)

target_sources(
    ttnn_op_full
    PRIVATE
        device/full_device_operation.cpp
        device/full_program_factory.cpp
        full.cpp
)

target_include_directories(ttnn_op_full PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_full
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_full LIBRARY COMPONENT tar)
