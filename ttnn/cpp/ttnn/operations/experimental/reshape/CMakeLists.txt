include(sources.cmake)

add_library(ttnn_op_experimental_reshape ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Reshape ALIAS ttnn_op_experimental_reshape)

tt_reuse_precompile_headers(ttnn_op_experimental_reshape TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_reshape)

set_target_properties(
    ttnn_op_experimental_reshape
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)
target_sources(
    ttnn_op_experimental_reshape
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_RESHAPE_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_RESHAPE_SRCS}
)

target_link_libraries(ttnn_op_experimental_reshape PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_reshape FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
