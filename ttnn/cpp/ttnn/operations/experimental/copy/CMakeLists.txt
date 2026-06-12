include(sources.cmake)

add_library(ttnn_op_experimental_copy ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Copy ALIAS ttnn_op_experimental_copy)

tt_reuse_precompile_headers(ttnn_op_experimental_copy TTNN::PCHLean)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_copy)

set_target_properties(
    ttnn_op_experimental_copy
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)
target_sources(
    ttnn_op_experimental_copy
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_COPY_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_COPY_SRCS}
)

target_link_libraries(ttnn_op_experimental_copy PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_copy FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
