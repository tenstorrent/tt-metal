add_library(ttnn_op_normalization ${LIB_TYPE})
add_library(TTNN::Ops::Normalization ALIAS ttnn_op_normalization)

target_precompile_headers(ttnn_op_normalization REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_normalization)
set_target_properties(
    ttnn_op_normalization
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            ""
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    batch_norm/device/kernels/*
    kernel_util/compute/*
    kernel_util/dataflow/*
    kernel_util/generic/*
)
target_sources(
    ttnn_op_normalization
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        batch_norm/batch_norm.cpp
        batch_norm/device/batch_norm_device_operation.cpp
        batch_norm/device/batch_norm_program_factory.cpp
        batch_norm/device/running_statistics_device_operation.cpp
        batch_norm/device/running_statistics_program_factory.cpp
        batch_norm/device/batch_norm_utils.cpp
)

target_include_directories(ttnn_op_normalization PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_normalization PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_normalization
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/normalization
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_normalization LIBRARY COMPONENT tar)
