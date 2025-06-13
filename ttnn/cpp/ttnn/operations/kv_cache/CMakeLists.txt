add_library(ttnn_op_kv_cache ${LIB_TYPE})
add_library(TTNN::Ops::KvCache ALIAS ttnn_op_kv_cache)

target_precompile_headers(ttnn_op_kv_cache REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_kv_cache)

target_sources(
    ttnn_op_kv_cache
    PRIVATE
        device/update_cache_op.cpp
        device/update_cache_op_multi_core.cpp
        kv_cache.cpp
)

target_include_directories(ttnn_op_kv_cache PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_kv_cache
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_kv_cache LIBRARY COMPONENT tar)
