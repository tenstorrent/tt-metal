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

# Kernels come from ttnn_op_index_fill (index_fill/device/kernels); no kernel file set here.
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

# Python bindings go on the shared `ttnn` module target; see index_fill/CMakeLists.txt.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_INDEX_FILL_NEW_NANOBIND_SRCS})
endif()
