add_library(ttnn_op_experimental_deepseek_prefill_extract ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeekPrefill::Extract ALIAS ttnn_op_experimental_deepseek_prefill_extract)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_extract REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_extract)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_extract
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_extract
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
        device/extract_device_operation.cpp
        device/extract_program_factory.cpp
        extract.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_prefill_extract PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_extract
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_extract
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/extract
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_extract LIBRARY COMPONENT tar)
