include(sources.cmake)

add_library(ttnn_op_experimental_kda_prepare_chunk_recurrence ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::KDA::PrepareChunkRecurrence
    ALIAS ttnn_op_experimental_kda_prepare_chunk_recurrence
)

tt_reuse_precompile_headers(ttnn_op_experimental_kda_prepare_chunk_recurrence TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_kda_prepare_chunk_recurrence)

set_target_properties(
    ttnn_op_experimental_kda_prepare_chunk_recurrence
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*.cpp)

target_sources(
    ttnn_op_experimental_kda_prepare_chunk_recurrence
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_KDA_PREPARE_CHUNK_RECURRENCE_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_KDA_PREPARE_CHUNK_RECURRENCE_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_kda_prepare_chunk_recurrence
    PRIVATE
        TT::Metalium
        TTNN::Ops::Experimental::KDA::Common
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_kda_prepare_chunk_recurrence
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_kda_prepare_chunk_recurrence FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_KDA_PREPARE_CHUNK_RECURRENCE_NANOBIND_SRCS})
endif()
