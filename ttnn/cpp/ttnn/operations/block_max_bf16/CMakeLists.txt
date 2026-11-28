include(sources.cmake)

add_library(ttnn_op_block_max_bf16 ${LIB_TYPE})
add_library(TTNN::Ops::BlockMaxBf16 ALIAS ttnn_op_block_max_bf16)

tt_reuse_precompile_headers(ttnn_op_block_max_bf16 TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_block_max_bf16)

set_target_properties(
    ttnn_op_block_max_bf16
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# The kernels and the args header they share with the program factory.
file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)

target_sources(
    ttnn_op_block_max_bf16
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_BLOCK_MAX_BF16_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_BLOCK_MAX_BF16_SRCS}
)

target_link_libraries(ttnn_op_block_max_bf16 PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_block_max_bf16
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/block_max_bf16
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_block_max_bf16 FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn` Python module target; the
# file list lives in sources.cmake. `ttnn` only exists when WITH_PYTHON_BINDINGS is on, hence the guard.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_BLOCK_MAX_BF16_NANOBIND_SRCS})
    # Built on its own, outside the `ttnn` target's unity batches: adding it to a batch shifts every later batch, which
    # put the bringup moe_block packages' copies of combine_plan_words.hpp (one namespace, three files) into one
    # translation unit. Kept out, the other files' batches stay as they were.
    set_source_files_properties(
        ${TTNN_OP_BLOCK_MAX_BF16_NANOBIND_SRCS}
        TARGET_DIRECTORY
            ttnn
        PROPERTIES
            SKIP_UNITY_BUILD_INCLUSION
                ON
    )
endif()
