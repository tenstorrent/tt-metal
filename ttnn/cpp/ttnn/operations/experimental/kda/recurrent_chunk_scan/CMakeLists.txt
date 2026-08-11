include(sources.cmake)

add_library(ttnn_op_experimental_kda_recurrent_chunk_scan ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::KDA::RecurrentChunkScan ALIAS ttnn_op_experimental_kda_recurrent_chunk_scan)

tt_reuse_precompile_headers(ttnn_op_experimental_kda_recurrent_chunk_scan TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_kda_recurrent_chunk_scan)

set_target_properties(
    ttnn_op_experimental_kda_recurrent_chunk_scan
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_kda_recurrent_chunk_scan
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_KDA_RECURRENT_CHUNK_SCAN_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_KDA_RECURRENT_CHUNK_SCAN_SRCS}
        ${TTNN_OP_EXPERIMENTAL_KDA_SHARED_SRCS}
)

target_link_libraries(ttnn_op_experimental_kda_recurrent_chunk_scan PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_kda_recurrent_chunk_scan FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_KDA_RECURRENT_CHUNK_SCAN_NANOBIND_SRCS})
endif()
