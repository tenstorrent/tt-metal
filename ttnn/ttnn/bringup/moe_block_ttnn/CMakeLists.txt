# The MoE block's ops (ttnn.experimental.moe_block): one library. common/ and every op directory own a sources.cmake
# that appends to the three lists below; an op adds its files there, never here. Kernels under <op>/device/kernels/
# are globbed.
set(TTNN_OP_BRINGUP_MOE_BLOCK_TTNN_API_HEADERS)
set(TTNN_OP_BRINGUP_MOE_BLOCK_TTNN_SRCS)
set(TTNN_OP_BRINGUP_MOE_BLOCK_TTNN_NANOBIND_SRCS)

include(common/sources.cmake)
include(moe_head/sources.cmake)
include(sparse_dispatch_2d/sources.cmake)
include(flat_routed_expert/sources.cmake)
include(routed_experts_combine/sources.cmake)
include(moe_tail/sources.cmake)
include(block/sources.cmake)

list(APPEND TTNN_OP_BRINGUP_MOE_BLOCK_TTNN_NANOBIND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/moe_block_nanobind.cpp)

add_library(ttnn_op_bringup_moe_block_ttnn ${LIB_TYPE})
add_library(TTNN::Ops::Bringup::MoeBlockTtnn ALIAS ttnn_op_bringup_moe_block_ttnn)

tt_reuse_precompile_headers(ttnn_op_bringup_moe_block_ttnn TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_bringup_moe_block_ttnn)
target_include_directories(ttnn_op_bringup_moe_block_ttnn PRIVATE ${FixmeOpAPIDir})

set_target_properties(
    ttnn_op_bringup_moe_block_ttnn
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    */device/kernels/*.cpp
    */device/kernels/*.hpp
    */device/kernels/*.h
    common/kernels/*.hpp
)
target_sources(
    ttnn_op_bringup_moe_block_ttnn
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${TTNN_OP_BRINGUP_MOE_BLOCK_TTNN_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_BRINGUP_MOE_BLOCK_TTNN_SRCS}
)

target_link_libraries(ttnn_op_bringup_moe_block_ttnn PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_bringup_moe_block_ttnn
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/ttnn/bringup/moe_block_ttnn
        COMPONENT ttnn-runtime
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_BRINGUP_MOE_BLOCK_TTNN_NANOBIND_SRCS})
endif()

# Compiled from ttnn/sources.cmake for the source op; the fork builds them in its own target.
target_sources(
    ttnn_op_bringup_moe_block_ttnn
    PRIVATE
        block/moe_block_arena.cpp
        common/args_table.cpp
        common/chip_program.cpp
        common/l1_budget.cpp
        common/mesh_routing.cpp
        common/moe_block_config.cpp
        common/moe_block_resources.cpp
        common/port_placement.cpp
        flat_routed_expert/device/flat_routed_expert_device_operation.cpp
        flat_routed_expert/device/flat_routed_expert_options.cpp
        flat_routed_expert/device/flat_routed_expert_plan.cpp
        flat_routed_expert/device/flat_routed_expert_program_factory.cpp
        flat_routed_expert/flat_routed_expert.cpp
        moe_head/device/head_engine_program.cpp
        moe_head/device/head_gather_program.cpp
        moe_head/device/head_model.cpp
        moe_head/device/head_plan.cpp
        moe_head/device/head_program.cpp
        moe_head/device/head_router_program.cpp
        moe_head/device/head_row_blocks.cpp
        moe_head/device/head_sizing.cpp
        moe_head/device/head_weights.cpp
        moe_head/device/head_workload.cpp
        moe_head/device/moe_head_device_operation.cpp
        moe_head/moe_head.cpp
        moe_tail/device/moe_tail_device_operation.cpp
        moe_tail/device/moe_tail_plan.cpp
        moe_tail/device/moe_tail_program_factory.cpp
        moe_tail/moe_tail.cpp
        routed_experts_combine/device/combine_host.cpp
        routed_experts_combine/device/combine_program.cpp
        routed_experts_combine/device/routed_experts_combine_device_operation.cpp
        routed_experts_combine/device/routed_experts_combine_program_factory.cpp
        routed_experts_combine/routed_experts_combine.cpp
        sparse_dispatch_2d/device/sparse_dispatch_2d_device_operation.cpp
        sparse_dispatch_2d/device/sparse_dispatch_2d_plan.cpp
        sparse_dispatch_2d/device/sparse_dispatch_2d_program_factory.cpp
        sparse_dispatch_2d/sparse_dispatch_2d.cpp
)
