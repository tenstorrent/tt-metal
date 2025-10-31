add_library(ttnn_op_experimental_unary_backward ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::UnaryBackward ALIAS ttnn_op_experimental_unary_backward)

target_precompile_headers(ttnn_op_experimental_unary_backward REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_unary_backward)

set_target_properties(
    ttnn_op_experimental_unary_backward
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels gelu_backward/device/kernels/*)

target_sources(
    ttnn_op_experimental_unary_backward
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES gelu_backward/gelu_backward.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        gelu_backward/device/gelu_backward_device_operation.cpp
        gelu_backward/device/gelu_backward_program_factory.cpp
        gelu_backward/gelu_backward.cpp
)

target_include_directories(ttnn_op_experimental_unary_backward PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_unary_backward PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_experimental_unary_backward
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/unary_backward
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_unary_backward LIBRARY COMPONENT tar)
