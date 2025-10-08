add_library(ttnn_op_eltwise_quantization ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Quantization ALIAS ttnn_op_eltwise_quantization)

target_precompile_headers(ttnn_op_eltwise_quantization REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_quantization)

target_sources(
    ttnn_op_eltwise_quantization
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES quantization.hpp
    PRIVATE
        quantization.cpp
)

target_include_directories(ttnn_op_eltwise_quantization PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_eltwise_quantization PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_eltwise_quantization
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/quantization
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_eltwise_quantization LIBRARY COMPONENT tar)
