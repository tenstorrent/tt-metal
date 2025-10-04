add_library(ttnn_op_eltwise_lazy ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Lazy ALIAS ttnn_op_eltwise_lazy)

target_precompile_headers(ttnn_op_eltwise_lazy REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_lazy)

target_sources(
    ttnn_op_eltwise_lazy
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES expression.hpp node.hpp operation.hpp param.hpp
    PRIVATE
        expression.cpp
)

target_include_directories(ttnn_op_eltwise_lazy PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_eltwise_lazy PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_eltwise_lazy
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/lazy
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_eltwise_lazy LIBRARY COMPONENT tar)
