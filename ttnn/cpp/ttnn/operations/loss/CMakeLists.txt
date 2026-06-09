include(sources.cmake)

add_library(ttnn_op_loss ${LIB_TYPE})
add_library(TTNN::Ops::Loss ALIAS ttnn_op_loss)

tt_reuse_precompile_headers(ttnn_op_loss TTNN::PCHLean)
TT_ENABLE_UNITY_BUILD(ttnn_op_loss)

set_target_properties(
    ttnn_op_loss
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_loss
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_LOSS_API_HEADERS}
    PRIVATE
        ${TTNN_OP_LOSS_SRCS}
)

target_link_libraries(ttnn_op_loss PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_loss FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
