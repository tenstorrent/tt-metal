add_library(ttnn_op_gather_new ${LIB_TYPE})
add_library(TTNN::Ops::GatherNew ALIAS ttnn_op_gather_new)

tt_reuse_precompile_headers(ttnn_op_gather_new TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_gather_new)

set_target_properties(
    ttnn_op_gather_new
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE gather_new_kernels CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../gather/device/kernels/*")

target_sources(
    ttnn_op_gather_new
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../gather
        FILES ${gather_new_kernels}
    PRIVATE
        gather_new.cpp
        device/gather_new_device_operation.cpp
        device/gather_new_program_factory.cpp
)

target_include_directories(ttnn_op_gather_new PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_gather_new
    PRIVATE
        TT::Metalium
        TTNN::Core
        TTNN::Ops::DataMovement
)

install(
    TARGETS
        ttnn_op_gather_new
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/data_movement/gather_new
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_gather_new LIBRARY COMPONENT tar)
