include(sources.cmake)

add_library(ttnn_op_experimental_kda_sigmoid_gated_rms_norm ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::KDA::SigmoidGatedRMSNorm ALIAS ttnn_op_experimental_kda_sigmoid_gated_rms_norm)

tt_reuse_precompile_headers(ttnn_op_experimental_kda_sigmoid_gated_rms_norm TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_kda_sigmoid_gated_rms_norm)

set_target_properties(
    ttnn_op_experimental_kda_sigmoid_gated_rms_norm
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*.cpp)

target_sources(
    ttnn_op_experimental_kda_sigmoid_gated_rms_norm
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_KDA_SIGMOID_GATED_RMS_NORM_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_KDA_SIGMOID_GATED_RMS_NORM_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_kda_sigmoid_gated_rms_norm
    PRIVATE
        TT::Metalium
        TTNN::Ops::Experimental::KDA::Common
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_kda_sigmoid_gated_rms_norm
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_kda_sigmoid_gated_rms_norm FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_KDA_SIGMOID_GATED_RMS_NORM_NANOBIND_SRCS})
endif()
