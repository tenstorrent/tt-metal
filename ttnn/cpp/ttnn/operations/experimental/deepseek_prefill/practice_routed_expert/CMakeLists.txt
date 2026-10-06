include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_practice_routed_expert ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::PracticeRoutedExpert
    ALIAS ttnn_op_experimental_deepseek_prefill_practice_routed_expert
)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_prefill_practice_routed_expert TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_practice_routed_expert)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_practice_routed_expert
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*.cpp)

target_sources(
    ttnn_op_experimental_deepseek_prefill_practice_routed_expert
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PRACTICE_ROUTED_EXPERT_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/practice_routed_expert_device_operation.cpp
        device/practice_routed_expert_program_factory.cpp
        practice_routed_expert.cpp
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_practice_routed_expert
    PRIVATE
        TT::Metalium
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_practice_routed_expert
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/practice_routed_expert
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_practice_routed_expert
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PRACTICE_ROUTED_EXPERT_NANOBIND_SRCS})
endif()
