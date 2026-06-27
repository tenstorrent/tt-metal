include(sources.cmake)

add_library(ttnn_op_math_fft ${LIB_TYPE})
add_library(TTNN::Ops::Math::FFT ALIAS ttnn_op_math_fft)

tt_reuse_precompile_headers(ttnn_op_math_fft TTNN::PCHLean)
TT_ENABLE_UNITY_BUILD(ttnn_op_math_fft)

target_sources(
    ttnn_op_math_fft
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_MATH_FFT_API_HEADERS}
    PRIVATE
        ${TTNN_OP_MATH_FFT_SRCS}
)

target_include_directories(ttnn_op_math_fft PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_math_fft PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_math_fft FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
