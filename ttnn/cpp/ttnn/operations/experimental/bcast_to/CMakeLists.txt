add_library(ttnn_op_experimental_bcast_to ${LIB_TYPE})
add_library(TT::NN::Ops::Experimental::BcastTo ALIAS ttnn_op_experimental_bcast_to)

target_precompile_headers(ttnn_op_experimental_bcast_to REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_bcast_to)

target_sources(
    ttnn_op_experimental_bcast_to
    PRIVATE
        bcast_to.cpp
        device/bcast_to_device_operation.cpp
        device/bcast_to_program_factory.cpp
        device/bcast_to_utils.cpp
)

target_include_directories(ttnn_op_experimental_bcast_to PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_bcast_to
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_experimental_bcast_to LIBRARY COMPONENT tar)
