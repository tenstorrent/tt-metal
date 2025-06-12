add_library(ttnn_op_point_to_point ${LIB_TYPE})
add_library(TTNN::Ops::PointToPoint ALIAS ttnn_op_point_to_point)

target_precompile_headers(ttnn_op_point_to_point REUSE_FROM TT::CommonPCH)

target_sources(
    ttnn_op_point_to_point
    PRIVATE
        device/host/point_to_point_device_op.cpp
        device/host/send_program_factory.cpp
        device/host/receive_program_factory.cpp
        point_to_point.cpp
)

target_include_directories(ttnn_op_point_to_point PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_point_to_point
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_point_to_point LIBRARY COMPONENT tar)
