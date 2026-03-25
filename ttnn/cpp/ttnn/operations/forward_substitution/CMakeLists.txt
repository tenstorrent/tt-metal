add_library(ttnn_op_forward_substitution ${LIB_TYPE})
add_library(TTNN::Ops::ForwardSubstitution ALIAS ttnn_op_forward_substitution)

tt_reuse_precompile_headers(ttnn_op_forward_substitution TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_forward_substitution)

set_target_properties(
    ttnn_op_forward_substitution
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_forward_substitution
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
        device/forward_substitution_device_operation.cpp
        device/forward_substitution_multi_core_factory.cpp
        forward_substitution.cpp
)

target_include_directories(ttnn_op_forward_substitution PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_forward_substitution
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_forward_substitution
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/forward_substitution
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_forward_substitution LIBRARY COMPONENT tar)
