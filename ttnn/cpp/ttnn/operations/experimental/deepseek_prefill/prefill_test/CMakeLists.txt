include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_prefill_test ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::PrefillTest
    ALIAS ttnn_op_experimental_deepseek_prefill_prefill_test
)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_prefill_prefill_test TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_prefill_test)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_prefill_test
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_deepseek_prefill_prefill_test
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PREFILL_TEST_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PREFILL_TEST_SRCS}
)

target_include_directories(ttnn_op_experimental_deepseek_prefill_prefill_test PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_deepseek_prefill_prefill_test PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_experimental_deepseek_prefill_prefill_test FILE_SET api COMPONENT ttnn-dev)

install(TARGETS ttnn_op_experimental_deepseek_prefill_prefill_test LIBRARY COMPONENT tar)
