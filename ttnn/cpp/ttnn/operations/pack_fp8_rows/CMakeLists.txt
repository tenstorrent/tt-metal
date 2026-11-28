include(sources.cmake)

add_library(ttnn_op_pack_fp8_rows ${LIB_TYPE})
add_library(TTNN::Ops::PackFp8Rows ALIAS ttnn_op_pack_fp8_rows)

tt_reuse_precompile_headers(ttnn_op_pack_fp8_rows TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_pack_fp8_rows)

set_target_properties(
    ttnn_op_pack_fp8_rows
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
    ttnn_op_pack_fp8_rows
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_PACK_FP8_ROWS_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_PACK_FP8_ROWS_SRCS}
)

target_link_libraries(ttnn_op_pack_fp8_rows PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_pack_fp8_rows
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/pack_fp8_rows
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_pack_fp8_rows FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn` Python module target; the
# file list lives in sources.cmake. `ttnn` only exists when WITH_PYTHON_BINDINGS is on, hence the guard.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_PACK_FP8_ROWS_NANOBIND_SRCS})
    # Built on its own, outside the `ttnn` target's unity batches: adding it to a batch shifts every later batch, which
    # put the bringup moe_block packages' copies of combine_plan_words.hpp (one namespace, three files) into one
    # translation unit. Kept out, the other files' batches stay as they were.
    set_source_files_properties(
        ${TTNN_OP_PACK_FP8_ROWS_NANOBIND_SRCS}
        TARGET_DIRECTORY
            ttnn
        PROPERTIES
            SKIP_UNITY_BUILD_INCLUSION
                ON
    )
endif()
