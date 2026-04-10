add_library(ttnn_op_experimental_turbo_quant ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::TurboQuant ALIAS ttnn_op_experimental_turbo_quant)

tt_reuse_precompile_headers(ttnn_op_experimental_turbo_quant TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_turbo_quant)

set_target_properties(
    ttnn_op_experimental_turbo_quant
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    device/kernels/*
    sdpa/kernels/*
)

target_sources(
    ttnn_op_experimental_turbo_quant
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
        turbo_quant.cpp
        device/turbo_quant_device_operation.cpp
        device/turbo_quant_program_factory.cpp
        sdpa/device/sdpa_tq_device_operation.cpp
        sdpa/device/sdpa_tq_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_turbo_quant PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_turbo_quant
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_turbo_quant
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/turbo_quant
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_turbo_quant LIBRARY COMPONENT tar)
