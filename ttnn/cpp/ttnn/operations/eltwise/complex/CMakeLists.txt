add_library(ttnn_op_eltwise_complex ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Complex ALIAS ttnn_op_eltwise_complex)

target_precompile_headers(ttnn_op_eltwise_complex REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_complex)

target_sources(
    ttnn_op_eltwise_complex
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES complex.hpp
    PRIVATE
        complex.cpp
)

target_include_directories(ttnn_op_eltwise_complex PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_eltwise_complex PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_eltwise_complex FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
