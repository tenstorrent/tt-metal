include(sources.cmake)

add_library(ttnn_op_core ${LIB_TYPE})
add_library(TTNN::Ops::Core ALIAS ttnn_op_core)

tt_reuse_precompile_headers(ttnn_op_core TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_core)

target_sources(
    ttnn_op_core
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_CORE_API_HEADERS}
    PRIVATE
        ${TTNN_OP_CORE_SRCS}
)

target_include_directories(ttnn_op_core PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_core
    PUBLIC
        TTNN::Core
        TT::Metalium
)

install(TARGETS ttnn_op_core FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
