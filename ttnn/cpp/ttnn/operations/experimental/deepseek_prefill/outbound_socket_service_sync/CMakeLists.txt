include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_outbound_socket_service_sync ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::D2dSocketSync
    ALIAS ttnn_op_experimental_deepseek_prefill_outbound_socket_service_sync
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_outbound_socket_service_sync REUSE_FROM TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_outbound_socket_service_sync)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_outbound_socket_service_sync
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_outbound_socket_service_sync
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_D2D_SOCKET_SYNC_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_D2D_SOCKET_SYNC_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_outbound_socket_service_sync
    PRIVATE
        TT::Metalium
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_outbound_socket_service_sync
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/outbound_socket_service_sync
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_outbound_socket_service_sync
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)
