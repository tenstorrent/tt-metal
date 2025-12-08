add_library(ttnn_op_experimental_dropout ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Dropout ALIAS ttnn_op_experimental_dropout)

target_precompile_headers(ttnn_op_experimental_dropout REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_dropout)

set_target_properties(
    ttnn_op_experimental_dropout
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_dropout
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES dropout.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/dropout_device_operation.cpp
        device/dropout_program_factory.cpp
        dropout.cpp
)

target_include_directories(ttnn_op_experimental_dropout PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_dropout PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_experimental_dropout
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/dropout
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_dropout LIBRARY COMPONENT tar)
