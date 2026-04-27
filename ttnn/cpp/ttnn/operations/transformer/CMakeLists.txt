add_library(ttnn_op_transformer ${LIB_TYPE})
add_library(TTNN::Ops::Transformer ALIAS ttnn_op_transformer)

tt_reuse_precompile_headers(ttnn_op_transformer TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_transformer)

set_target_properties(
    ttnn_op_transformer
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)
# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels)
target_sources(
    ttnn_op_transformer
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        concatenate_heads/concatenate_heads.cpp
        split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.cpp
)

target_include_directories(ttnn_op_transformer PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_transformer
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_transformer
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/transformer
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_transformer LIBRARY COMPONENT tar)
