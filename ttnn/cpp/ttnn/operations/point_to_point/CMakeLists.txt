get_filename_component(OP_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

add_library(ttnn_op_point_to_point ${LIB_TYPE})
add_library(TTNN::Ops::PointToPoint ALIAS ttnn_op_point_to_point)

target_precompile_headers(ttnn_op_point_to_point REUSE_FROM TT::CommonPCH)

set_target_properties(
    ttnn_op_point_to_point
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_point_to_point
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
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

install(
    TARGETS
        ttnn_op_point_to_point
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/${OP_NAME}
        COMPONENT ttnn-runtime
)
install(TARGETS ttnn_op_point_to_point LIBRARY COMPONENT tar)
