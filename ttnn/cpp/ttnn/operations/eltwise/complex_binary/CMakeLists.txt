add_library(ttnn_op_eltwise_complex_binary ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Complex::Binary ALIAS ttnn_op_eltwise_complex_binary)

target_precompile_headers(ttnn_op_eltwise_complex_binary REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_complex_binary)

target_sources(
    ttnn_op_eltwise_complex_binary
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES device/complex_binary_op.hpp
    PRIVATE
        device/complex_binary_op.cpp
)

target_include_directories(ttnn_op_eltwise_complex_binary PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_eltwise_complex_binary PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_eltwise_complex_binary
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/complex_binary
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_eltwise_complex_binary LIBRARY COMPONENT tar)
