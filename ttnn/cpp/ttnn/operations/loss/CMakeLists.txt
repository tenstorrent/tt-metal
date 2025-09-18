add_library(ttnn_op_loss ${LIB_TYPE})
add_library(TTNN::Ops::Loss ALIAS ttnn_op_loss)

target_precompile_headers(ttnn_op_loss REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_loss)

target_sources(ttnn_op_loss PRIVATE loss.cpp)

target_include_directories(ttnn_op_loss PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_loss
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_loss LIBRARY COMPONENT tar)
