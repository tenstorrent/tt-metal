add_library(ttnn_op_experimental_test ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Test ALIAS ttnn_op_experimental_test)

target_precompile_headers(ttnn_op_experimental_test REUSE_FROM TT::CommonPCH)

set_target_properties(
    ttnn_op_experimental_test
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels hang_device/device/kernels/*)

target_sources(
    ttnn_op_experimental_test
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
        hang_device/hang_device_operation.cpp
)

target_include_directories(ttnn_op_experimental_test PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_test
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_test
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/test
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_test LIBRARY COMPONENT tar)
