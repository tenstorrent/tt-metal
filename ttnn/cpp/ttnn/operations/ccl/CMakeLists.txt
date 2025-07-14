add_library(ttnn_op_ccl ${LIB_TYPE})
add_library(TTNN::Ops::CCL ALIAS ttnn_op_ccl)

target_precompile_headers(ttnn_op_ccl REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_ccl)

target_sources(
    ttnn_op_ccl
    PRIVATE
        # Common
        erisc_datamover_builder_helper.cpp
        ccl_op_fusion.cpp
        ccl_common.cpp
        ccl_host_datastructures.cpp
        common/types/ccl_types_args_emitters.cpp
        common/host/command_backend_runtime_args_overrider.cpp
        common/uops/ccl_command.cpp
        common/uops/command_lowering.cpp
        common/uops/ccl_host_commands.cpp
        common/host/ccl_worker_builder.cpp
        common/host/ccl_command_stream_builders.cpp
        common/host/moe_utils.cpp
        # Ops
        all_gather/all_gather.cpp
        all_gather/device/all_gather_op.cpp
        all_gather/device/multi_core/all_gather_op_multi_core.cpp
        all_to_all_combine/all_to_all_combine.cpp
        all_to_all_combine/device/all_to_all_combine_device_operation.cpp
        all_to_all_combine/device/all_to_all_combine_program_factory.cpp
        all_to_all_dispatch/all_to_all_dispatch.cpp
        all_to_all_dispatch/device/all_to_all_dispatch_device_operation.cpp
        all_to_all_dispatch/device/all_to_all_dispatch_program_factory.cpp
        reduce_scatter/device/host/reduce_scatter_full_worker_grid.cpp
        reduce_scatter/device/reduce_scatter_op.cpp
        reduce_scatter/reduce_scatter.cpp
        reduce_scatter/host/reduce_scatter_worker_builder.cpp
        reduce_scatter/host/reduce_scatter_common.cpp
        barrier/device/host/barrier_full_worker_grid.cpp
        barrier/device/barrier_op.cpp
        barrier/barrier.cpp
)

target_include_directories(ttnn_op_ccl PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_ccl
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_ccl LIBRARY COMPONENT tar)
