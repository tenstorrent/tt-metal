add_library(ttnn_op_experimental_deepseek_prefill_insert ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeekPrefill::Insert ALIAS ttnn_op_experimental_deepseek_prefill_insert)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_insert REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_insert)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_insert
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_insert
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
        device/insert_device_operation.cpp
        device/insert_program_factory.cpp
        insert.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_prefill_insert PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_insert
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_insert
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/insert
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_insert LIBRARY COMPONENT tar)
