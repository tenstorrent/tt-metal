add_library(ttnn_op_eltwise_unary_backward ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Unary::Backward ALIAS ttnn_op_eltwise_unary_backward)

target_precompile_headers(ttnn_op_eltwise_unary_backward REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_unary_backward)

target_sources(
    ttnn_op_eltwise_unary_backward
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES unary_backward.hpp
    PRIVATE
        unary_backward.cpp
)

target_include_directories(ttnn_op_eltwise_unary_backward PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_eltwise_unary_backward PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_eltwise_unary_backward
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/unary_backward
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_eltwise_unary_backward LIBRARY COMPONENT tar)
