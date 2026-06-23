include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_h2d_socket_sync ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::H2dSocketSync
    ALIAS ttnn_op_experimental_deepseek_prefill_h2d_socket_sync
)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_prefill_h2d_socket_sync TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_h2d_socket_sync)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_h2d_socket_sync
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_h2d_socket_sync
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_H2D_SOCKET_SYNC_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_H2D_SOCKET_SYNC_SRCS}
)

target_include_directories(ttnn_op_experimental_deepseek_prefill_h2d_socket_sync PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_deepseek_prefill_h2d_socket_sync PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_h2d_socket_sync
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/h2d_socket_sync
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_h2d_socket_sync LIBRARY COMPONENT tar)
