add_library(ttnn_op_examples ${LIB_TYPE})
add_library(TTNN::Ops::Examples ALIAS ttnn_op_examples)

target_precompile_headers(ttnn_op_examples REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_examples)

set_target_properties(
    ttnn_op_examples
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    example/device/kernels/*
    example_multiple_return/device/kernels/*
)

target_sources(
    ttnn_op_examples
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
        example/device/example_device_operation.cpp
        example/device/multi_core_program_factory.cpp
        example/device/single_core_program_factory.cpp
        example_multiple_return/device/example_multiple_return_device_operation.cpp
        example_multiple_return/device/single_core_program_factory.cpp
        example_multiple_return/example_multiple_return.cpp
)

target_include_directories(ttnn_op_examples PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_examples
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_examples
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/examples
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_examples LIBRARY COMPONENT tar)
