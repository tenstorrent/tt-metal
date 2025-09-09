add_library(ttnn_op_kv_cache ${LIB_TYPE})
add_library(TTNN::Ops::KvCache ALIAS ttnn_op_kv_cache)

target_precompile_headers(ttnn_op_kv_cache REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_kv_cache)
set_target_properties(
    ttnn_op_kv_cache
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_kv_cache
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES kv_cache.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
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

install(
    TARGETS
        ttnn_op_kv_cache
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/kv_cache
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_kv_cache LIBRARY COMPONENT tar)
