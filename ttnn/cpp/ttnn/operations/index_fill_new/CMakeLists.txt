include(sources.cmake)

add_library(ttnn_op_index_fill_new ${LIB_TYPE})
add_library(TTNN::Ops::IndexFillNew ALIAS ttnn_op_index_fill_new)

tt_reuse_precompile_headers(ttnn_op_index_fill_new TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_index_fill_new)

set_target_properties(
    ttnn_op_index_fill_new
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Phase-1 descriptor-migration scaffolding for index_fill (#42392). The kernels are reused
# unchanged from ttnn/cpp/ttnn/operations/index_fill/device/kernels/ (installed by
# ttnn_op_index_fill), so this target ships no kernel file set of its own.
target_sources(
    ttnn_op_index_fill_new
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_INDEX_FILL_NEW_API_HEADERS}
    PRIVATE
        ${TTNN_OP_INDEX_FILL_NEW_SRCS}
)

target_link_libraries(ttnn_op_index_fill_new PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_index_fill_new FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings registered on the shared `ttnn` module target (see index_fill/CMakeLists.txt
# for the rationale). Exposes ttnn.index_fill_new so the existing pytest suite can be pointed at
# the descriptor path during Phase 2.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_INDEX_FILL_NEW_NANOBIND_SRCS})
endif()
