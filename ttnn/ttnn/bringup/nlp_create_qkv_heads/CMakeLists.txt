include(sources.cmake)

add_library(ttnn_op_bringup_nlp_create_qkv_heads ${LIB_TYPE})
add_library(TTNN::Ops::Bringup::NlpCreateQkvHeads ALIAS ttnn_op_bringup_nlp_create_qkv_heads)

tt_reuse_precompile_headers(ttnn_op_bringup_nlp_create_qkv_heads TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_bringup_nlp_create_qkv_heads)
target_include_directories(
    ttnn_op_bringup_nlp_create_qkv_heads
    PRIVATE
        ${FixmeOpAPIDir}
        ${FixmeOpIncDirs}
)

set_target_properties(
    ttnn_op_bringup_nlp_create_qkv_heads
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
    device/kernels/*.h
)

target_sources(
    ttnn_op_bringup_nlp_create_qkv_heads
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${TTNN_OP_BRINGUP_NLP_CREATE_QKV_HEADS_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_BRINGUP_NLP_CREATE_QKV_HEADS_SRCS}
)

target_link_libraries(ttnn_op_bringup_nlp_create_qkv_heads PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_bringup_nlp_create_qkv_heads
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/ttnn/bringup/nlp_create_qkv_heads
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_bringup_nlp_create_qkv_heads FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_BRINGUP_NLP_CREATE_QKV_HEADS_NANOBIND_SRCS})
endif()
