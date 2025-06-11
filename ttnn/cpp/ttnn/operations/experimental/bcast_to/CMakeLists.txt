add_library(ttnn_op_experimental_bcast_to ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::BcastTo ALIAS ttnn_op_experimental_bcast_to)

target_precompile_headers(ttnn_op_experimental_bcast_to REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_bcast_to)
set_target_properties(
    ttnn_op_experimental_bcast_to
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

target_sources(
    ttnn_op_experimental_bcast_to
    PRIVATE
        bcast_to.cpp
        device/bcast_to_device_operation.cpp
        device/bcast_to_program_factory.cpp
        device/bcast_to_utils.cpp
)

file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_experimental_bcast_to
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
)

target_include_directories(ttnn_op_experimental_bcast_to PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_bcast_to
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_bcast_to
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/bcast_to
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_bcast_to LIBRARY COMPONENT tar)
