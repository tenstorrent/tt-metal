add_library(ttnn_op_transformer ${LIB_TYPE})
add_library(TTNN::Ops::Transformer ALIAS ttnn_op_transformer)

target_precompile_headers(ttnn_op_transformer REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_transformer)

set_target_properties(
    ttnn_op_transformer
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)
# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels sdpa_decode/device/kernels/*)
target_sources(
    ttnn_op_transformer
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        attention_softmax/attention_softmax.cpp
        concatenate_heads/concatenate_heads.cpp
        sdpa/device/ring_fusion.cpp
        sdpa/device/joint_sdpa_op.cpp
        sdpa/device/joint_sdpa_program_factory.cpp
        sdpa/device/ring_joint_sdpa_op.cpp
        sdpa/device/ring_joint_sdpa_program_factory.cpp
        sdpa/device/sdpa_op.cpp
        sdpa/device/sdpa_program_factory.cpp
        sdpa/sdpa.cpp
        sdpa_decode/device/sdpa_decode_op.cpp
        sdpa_decode/device/sdpa_decode_program_factory.cpp
        sdpa_decode/sdpa_decode.cpp
        sdpa_windowed/device/sdpa_windowed_op.cpp
        sdpa_windowed/device/sdpa_windowed_program_factory.cpp
        sdpa_windowed/sdpa_windowed.cpp
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
