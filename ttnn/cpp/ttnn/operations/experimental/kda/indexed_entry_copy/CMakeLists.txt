include(sources.cmake)

add_library(ttnn_op_experimental_kda_indexed_entry_copy ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::KDA::IndexedEntryCopy ALIAS ttnn_op_experimental_kda_indexed_entry_copy)

tt_reuse_precompile_headers(ttnn_op_experimental_kda_indexed_entry_copy TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_kda_indexed_entry_copy)

set_target_properties(
    ttnn_op_experimental_kda_indexed_entry_copy
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)

target_sources(
    ttnn_op_experimental_kda_indexed_entry_copy
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_KDA_INDEXED_ENTRY_COPY_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_KDA_INDEXED_ENTRY_COPY_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_kda_indexed_entry_copy
    PRIVATE
        TT::Metalium
        TTNN::Ops::Experimental::KDA::Common
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_kda_indexed_entry_copy
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/kda/indexed_entry_copy
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_kda_indexed_entry_copy FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_KDA_INDEXED_ENTRY_COPY_NANOBIND_SRCS})
endif()
