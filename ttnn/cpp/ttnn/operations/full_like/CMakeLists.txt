add_library(ttnn_op_full_like ${LIB_TYPE})
add_library(TTNN::Ops::FullLike ALIAS ttnn_op_full_like)

target_precompile_headers(ttnn_op_full_like REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_full_like)

target_sources(
    ttnn_op_full_like
    PRIVATE
        device/full_like_device_operation.cpp
        device/full_like_factory.cpp
        full_like.cpp
)

target_include_directories(ttnn_op_full_like PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_full_like
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_full_like LIBRARY COMPONENT tar)
