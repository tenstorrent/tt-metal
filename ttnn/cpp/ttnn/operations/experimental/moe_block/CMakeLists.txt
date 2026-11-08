# The MoE block's ops (ttnn.experimental.moe_block): one library. common/ and every op directory own a sources.cmake
# that appends to the three lists below; an op adds its files there, never here. Kernels under <op>/device/kernels/
# are globbed.
set(TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS)
set(TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS)
set(TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS)

include(common/sources.cmake)
include(fused_head/sources.cmake)
include(sparse_dispatch_2d/sources.cmake)
include(flat_routed_expert/sources.cmake)
include(routed_experts_combine/sources.cmake)
include(fused_tail/sources.cmake)
include(block/sources.cmake)

list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/moe_block_nanobind.cpp)

add_library(ttnn_op_experimental_moe_block ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::MoEBlock ALIAS ttnn_op_experimental_moe_block)

tt_reuse_precompile_headers(ttnn_op_experimental_moe_block TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_moe_block)

set_target_properties(
    ttnn_op_experimental_moe_block
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    */device/kernels/*.cpp
    */device/kernels/*.hpp
    */device/kernels/*.h
)
target_sources(
    ttnn_op_experimental_moe_block
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS}
)

target_link_libraries(ttnn_op_experimental_moe_block PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_moe_block
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/moe_block
        COMPONENT ttnn-runtime
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS})
endif()
