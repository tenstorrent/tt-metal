add_library(ttnn_op_eltwise_expression ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Expression ALIAS ttnn_op_eltwise_expression)

target_precompile_headers(ttnn_op_eltwise_expression REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_expression)

target_sources(
    ttnn_op_eltwise_expression
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES expression.hpp node.hpp operation.hpp operations.hpp overload.hpp param.hpp
    PRIVATE
        expression.cpp
        operations.cpp
)

target_include_directories(ttnn_op_eltwise_expression PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_eltwise_expression PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_eltwise_expression
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/expression
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_eltwise_expression LIBRARY COMPONENT tar)
