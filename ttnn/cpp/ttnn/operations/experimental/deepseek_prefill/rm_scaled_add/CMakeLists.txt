add_library(ttnn_op_experimental_rm_scaled_add ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::RmScaledAdd ALIAS ttnn_op_experimental_rm_scaled_add)

target_precompile_headers(ttnn_op_experimental_rm_scaled_add REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_rm_scaled_add)

set_target_properties(
    ttnn_op_experimental_rm_scaled_add
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_rm_scaled_add
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
        device/rm_scaled_add_device_operation.cpp
        device/rm_scaled_add_program_factory.cpp
        rm_scaled_add.cpp
)

target_include_directories(ttnn_op_experimental_rm_scaled_add PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_rm_scaled_add
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_rm_scaled_add
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rm_scaled_add
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_rm_scaled_add LIBRARY COMPONENT tar)
