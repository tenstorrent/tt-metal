include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_sharded_rmsnorm ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeek::ShardedRmsNorm ALIAS ttnn_op_experimental_deepseek_sharded_rmsnorm)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_sharded_rmsnorm TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_sharded_rmsnorm)

set_target_properties(
    ttnn_op_experimental_deepseek_sharded_rmsnorm
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_sharded_rmsnorm
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_SHARDED_RMSNORM_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_SHARDED_RMSNORM_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_sharded_rmsnorm PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_deepseek_sharded_rmsnorm
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek/sharded_rmsnorm
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_deepseek_sharded_rmsnorm FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_SHARDED_RMSNORM_NANOBIND_SRCS})
endif()
