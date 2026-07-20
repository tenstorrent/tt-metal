include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_dispatch_tilize ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::DispatchTilize
    ALIAS ttnn_op_experimental_deepseek_prefill_dispatch_tilize
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_dispatch_tilize REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_dispatch_tilize)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_dispatch_tilize
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*.cpp)

target_sources(
    ttnn_op_experimental_deepseek_prefill_dispatch_tilize
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DISPATCH_TILIZE_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DISPATCH_TILIZE_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_prefill_dispatch_tilize PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_dispatch_tilize
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_tilize
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_dispatch_tilize
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DISPATCH_TILIZE_NANOBIND_SRCS})
endif()
