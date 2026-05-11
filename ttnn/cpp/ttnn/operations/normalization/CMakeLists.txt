include(sources.cmake)

add_library(ttnn_op_normalization ${LIB_TYPE})
add_library(TTNN::Ops::Normalization ALIAS ttnn_op_normalization)

tt_reuse_precompile_headers(ttnn_op_normalization TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_normalization)
set_target_properties(
    ttnn_op_normalization
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    batch_norm/device/kernels/*
    groupnorm/device/kernels/*
    layernorm/device/kernels/*
    layernorm_distributed/device/kernels/*
    rmsnorm_distributed/device/kernels/*
    softmax/device/kernels/*
    kernel_util/compute/*
    kernel_util/dataflow/*
    kernel_util/generic/*
)
target_sources(
    ttnn_op_normalization
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_NORMALIZATION_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_NORMALIZATION_SRCS}
)

target_include_directories(ttnn_op_normalization PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_normalization PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_normalization
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/normalization
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_normalization LIBRARY COMPONENT tar)
