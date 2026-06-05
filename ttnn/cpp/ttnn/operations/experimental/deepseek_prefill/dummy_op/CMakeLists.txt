add_library(ttnn_op_experimental_deepseek_prefill_dummy_op ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeekPrefill::DummyOp ALIAS ttnn_op_experimental_deepseek_prefill_dummy_op)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_dummy_op REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_dummy_op)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_dummy_op
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_dummy_op
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
        device/dummy_op_device_operation.cpp
        device/dummy_op_program_factory.cpp
        dummy_op.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_prefill_dummy_op PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_dummy_op
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_dummy_op
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dummy_op
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_dummy_op LIBRARY COMPONENT tar)
