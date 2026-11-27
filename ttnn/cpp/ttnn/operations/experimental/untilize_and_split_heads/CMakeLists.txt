include(sources.cmake)

add_library(ttnn_op_experimental_untilize_and_split_heads ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::UntilizeAndSplitHeads ALIAS ttnn_op_experimental_untilize_and_split_heads)

tt_reuse_precompile_headers(ttnn_op_experimental_untilize_and_split_heads TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_untilize_and_split_heads)
set_target_properties(
    ttnn_op_experimental_untilize_and_split_heads
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_experimental_untilize_and_split_heads
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_UNTILIZE_AND_SPLIT_HEADS_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_UNTILIZE_AND_SPLIT_HEADS_SRCS}
)

target_include_directories(ttnn_op_experimental_untilize_and_split_heads PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_untilize_and_split_heads PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_experimental_untilize_and_split_heads LIBRARY COMPONENT tar)

install(
    TARGETS
        ttnn_op_experimental_untilize_and_split_heads
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/untilize_and_split_heads
        COMPONENT ttnn-runtime
)

# The nanobind source goes on the shared `ttnn` Python extension target, which
# exists only when WITH_PYTHON_BINDINGS is on.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_UNTILIZE_AND_SPLIT_HEADS_NANOBIND_SRCS})
endif()
