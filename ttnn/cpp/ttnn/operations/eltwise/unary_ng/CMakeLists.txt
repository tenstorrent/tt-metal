add_library(ttnn_op_eltwise_unary_ng ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::UnaryNG ALIAS ttnn_op_eltwise_unary_ng)

target_precompile_headers(ttnn_op_eltwise_unary_ng REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_unary_ng)
set_target_properties(
    ttnn_op_eltwise_unary_ng
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Unary NG compute kernels (runtime block count; single kernel like binary_ng)
file(GLOB_RECURSE unary_ng_kernels device/kernels/*)

target_sources(
    ttnn_op_eltwise_unary_ng
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES device/unary_ng_device_operation.hpp common/unary_ng_utils.hpp common/unary_ng_op_utils.hpp unary_ng.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${unary_ng_kernels}
    PRIVATE
        device/unary_ng_device_operation.cpp
        device/unary_ng_program_factory.cpp
        common/unary_ng_utils.cpp
        common/unary_ng_op_utils.cpp
        unary_ng.cpp
)

target_include_directories(ttnn_op_eltwise_unary_ng PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_unary_ng
    PRIVATE
        TT::Metalium
        TTNN::Core
        TTNN::Ops::Eltwise::Unary
)

install(
    TARGETS
        ttnn_op_eltwise_unary_ng
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/unary_ng
        COMPONENT ttnn-runtime
    LIBRARY
        COMPONENT tar
)
