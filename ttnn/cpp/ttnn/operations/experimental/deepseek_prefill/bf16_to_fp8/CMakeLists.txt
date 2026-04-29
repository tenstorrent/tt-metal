add_library(ttnn_op_experimental_deepseek_prefill_bf16_to_fp8 ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeekPrefill::Bf16ToFp8 ALIAS ttnn_op_experimental_deepseek_prefill_bf16_to_fp8)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_bf16_to_fp8 REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_bf16_to_fp8)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_bf16_to_fp8
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_bf16_to_fp8
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
        device/bf16_to_fp8_device_operation.cpp
        device/bf16_to_fp8_program_factory.cpp
        bf16_to_fp8.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_prefill_bf16_to_fp8 PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_bf16_to_fp8
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_bf16_to_fp8
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/bf16_to_fp8
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_bf16_to_fp8 LIBRARY COMPONENT tar)
