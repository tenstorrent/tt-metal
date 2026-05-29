include(sources.cmake)

add_library(ttnn_op_rand ${LIB_TYPE})
add_library(TTNN::Ops::Rand ALIAS ttnn_op_rand)

tt_reuse_precompile_headers(ttnn_op_rand TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_rand)

# Verify only the api FILE_SET
set_target_properties(
    ttnn_op_rand
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_rand
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_RAND_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_RAND_SRCS}
)

target_include_directories(ttnn_op_rand PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_rand PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_rand FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

install(
    TARGETS
        ttnn_op_rand
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/rand
        COMPONENT ttnn-runtime
)
