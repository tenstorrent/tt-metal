include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_moe_grouped_topk ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::MoeGroupedTopk
    ALIAS ttnn_op_experimental_deepseek_prefill_moe_grouped_topk
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_moe_grouped_topk REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_moe_grouped_topk)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_moe_grouped_topk
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_moe_grouped_topk
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
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_GROUPED_TOPK_SRCS}
)

target_include_directories(ttnn_op_experimental_deepseek_prefill_moe_grouped_topk PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_moe_grouped_topk
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_moe_grouped_topk
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_moe_grouped_topk LIBRARY COMPONENT tar)
