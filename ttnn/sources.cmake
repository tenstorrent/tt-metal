# Source file lists for the ttnn top-level targets.
# Module owners should update this file when adding/removing/renaming source files.
# Build logic lives in CMakeLists.txt — keep this file free of logic.

####################################################################################################
# ttnn_core PRIVATE sources
####################################################################################################

set(TTNN_CORE_SRCS
    core/async_runtime.cpp
    core/cluster.cpp
    core/config.cpp
    core/core.cpp
    core/device.cpp
    core/device_operation_detail.cpp
    cpp/tools/profiler/op_profiler_json.cpp
    core/distributed/api.cpp
    core/distributed/distributed_tensor.cpp
    core/distributed/distribution_mode.cpp
    core/distributed/host_ccl.cpp
    core/distributed/bidirectional_fabric_socket.cpp
    core/distributed/create_socket.cpp
    core/distributed/fabric_socket.cpp
    core/distributed/mpi_socket.cpp
    core/events.cpp
    core/global_circular_buffer.cpp
    core/global_semaphore.cpp
    core/operation.cpp
    core/graph/graph_processor.cpp
    core/graph/graph_trace_utils.cpp
    core/graph/levelized_graph.cpp
    core/up_front_compile.cpp
    core/reports.cpp
    core/tensor/flatbuffer/tensor_flatbuffer.cpp
    core/tensor/flatbuffer/tensor_spec_flatbuffer.cpp
    core/tensor/flatbuffer/overlapped_tensor_flatbuffer.cpp
    core/tensor/overlapped_serialization.cpp
    core/tensor/host_buffer/functions.cpp
    core/tensor/serialization.cpp
    core/tensor/storage.cpp
    core/tensor/tensor.cpp
    core/tensor/tensor_attributes.cpp
    core/tensor/tensor_impl.cpp
    core/tensor/tensor_ops.cpp
    core/services/h2d_socket_service.cpp
    core/services/d2h_socket_service.cpp
    core/tensor/d2d_stream_service.cpp
    cpp/ttnn/operations/experimental/core_subset_write/copy_to_device_filtered.cpp
    core/tensor/tensor_utils.cpp
    core/tensor/unit_mesh/unit_mesh_utils.cpp
    core/tensor/xtensor/partition.cpp
    core/tensor/to_string.cpp
    core/tensor/py_to_tt_tensor.cpp
)

####################################################################################################
# ttnncpp PRIVATE sources
####################################################################################################

set(TTNNCPP_SRCS
    # FIXME: Move these out to appropriate sub targets
    cpp/ttnn/operations/compute_throttle_utils.cpp
    cpp/ttnn/operations/trace.cpp
    cpp/ttnn/graph/capture_program_config_registry.cpp
    cpp/ttnn/operations/ccl/sharding_addrgen_helper.cpp
    cpp/ttnn/operations/generic/generic_op.cpp
    cpp/ttnn/operations/generic/device/generic_op_program_factory.cpp
    cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
    cpp/ttnn/operations/experimental/fusion/device/fusion_dispatch_op_device_operation.cpp
    cpp/ttnn/operations/data_movement/reshape_view/reshape_common.cpp
    cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_device_operation.cpp
    cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_program_factory.cpp
    cpp/ttnn/operations/experimental/ccl/rms_allgather/rms_allgather.cpp
    cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/dit_fused_distributed_rmsnorm.cpp
    cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/dit_fused_distributed_rmsnorm_device_operation.cpp
    cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/dit_fused_distributed_rmsnorm_program_factory.cpp
    cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/dispatch.cpp
    cpp/ttnn/operations/experimental/deepseek_prefill/combine/combine.cpp
    cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/routed_expert_ffn_common.cpp
    cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/routed_expert_ffn_wh.cpp
    cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/routed_expert_ffn_bh.cpp
    cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/unified_routed_expert_ffn.cpp
    cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_device_operation.cpp
    cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp
    cpp/ttnn/operations/experimental/test/hang_device/hang_device_program_factory.cpp
    cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.cpp
    cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_post_all_gather.cpp
    cpp/ttnn/operations/copy/typecast/device/typecast_device_op.cpp
    cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp
    cpp/ttnn/operations/copy/typecast/device/typecast_rm_chunked_program_factory.cpp
    cpp/ttnn/operations/copy/typecast/device/typecast_sharded_program_factory.cpp
    cpp/ttnn/operations/copy/typecast/typecast.cpp
)

####################################################################################################
# Python binding sources (nanobind)
####################################################################################################

set(TTNN_SRC_PYBIND
    core/distributed/distributed_nanobind.cpp
    core/graph/graph_nanobind.cpp
    cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/routed_expert_ffn_nanobind.cpp
    cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/unified_routed_expert_ffn_nanobind.cpp
    cpp/ttnn/operations/experimental/experimental_nanobind.cpp
    cpp/ttnn/operations/experimental/fusion/fusion_dispatch_op_nanobind.cpp
    cpp/ttnn/operations/generic/generic_op_nanobind.cpp
    # ttnn-nanobind core files (appended in original CMakeLists.txt)
    cpp/ttnn-nanobind/__init__.cpp
    cpp/ttnn-nanobind/activation.cpp
    cpp/ttnn-nanobind/cluster.cpp
    cpp/ttnn-nanobind/core.cpp
    cpp/ttnn-nanobind/device.cpp
    cpp/ttnn-nanobind/events.cpp
    cpp/ttnn-nanobind/fabric.cpp
    cpp/ttnn-nanobind/disaggregation.cpp
    cpp/ttnn/experimental/disaggregation/tensor_helpers.cpp
    cpp/ttnn-nanobind/global_circular_buffer.cpp
    cpp/ttnn-nanobind/global_semaphore.cpp
    cpp/ttnn-nanobind/hd_socket.cpp
    cpp/ttnn-nanobind/d2d_stream_service.cpp
    cpp/ttnn-nanobind/h2d_stream_service.cpp
    cpp/ttnn-nanobind/d2h_stream_service.cpp
    cpp/ttnn-nanobind/counter_channel.cpp
    cpp/ttnn-nanobind/mesh_socket.cpp
    cpp/ttnn-nanobind/profiler.cpp
    cpp/ttnn-nanobind/program_descriptors.cpp
    cpp/ttnn-nanobind/pytensor.cpp
    cpp/ttnn-nanobind/reports.cpp
    cpp/ttnn-nanobind/tensor.cpp
    cpp/ttnn-nanobind/types.cpp
    cpp/ttnn-nanobind/bfp_utils.cpp
    cpp/ttnn-nanobind/operations/copy.cpp
    cpp/ttnn-nanobind/operations/core.cpp
    cpp/ttnn-nanobind/operations/trace.cpp
    cpp/ttnn-nanobind/tensor_accessor_args.cpp
    cpp/ttnn-nanobind/pipeline_module_nanobind.cpp
)

# experimental/ccl/'s, point_to_point's, and debug/'s own nanobind sources
# are registered directly on the `ttnn` target from those ops' own
# CMakeLists.txt — not listed here.

####################################################################################################
# ttnn_core FILE_SET jit_api headers
####################################################################################################

set(TTNN_CORE_JIT_API_HEADERS
    api/ttnn/tensor/layout/layout.hpp
    cpp/ttnn/kernel/compute/bmm_tilize_untilize.cpp
    cpp/ttnn/kernel/compute/eltwise_copy.cpp
    cpp/ttnn/kernel/compute/moreh_common.hpp
    cpp/ttnn/kernel/compute/tilize.cpp
    cpp/ttnn/kernel/compute/transpose_wh.cpp
    cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp
    cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp
    cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp
    cpp/ttnn/kernel/dataflow/moreh_common.hpp
    cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp
    cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp
    cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp
    cpp/ttnn/kernel/kernel_common_utils.hpp
    cpp/ttnn/kernel/kernel_utils.hpp
)

####################################################################################################
# ttnncpp FILE_SET api headers
####################################################################################################

set(TTNNCPP_API_HEADERS
    api/tools/profiler/op_profiler.hpp
    api/tools/profiler/op_profiler_serialize.hpp
    api/ttnn/async_runtime.hpp
    api/ttnn/cluster.hpp
    api/ttnn/common/constants.hpp
    api/ttnn/common/guard.hpp
    api/ttnn/common/queue_id.hpp
    api/ttnn/config.hpp
    api/ttnn/core.hpp
    api/ttnn/device.hpp
    api/ttnn/device_operation.hpp
    api/ttnn/device_operation_detail.hpp
    api/ttnn/distributed/api.hpp
    api/ttnn/distributed/bidirectional_fabric_socket.hpp
    api/ttnn/distributed/create_socket.hpp
    api/ttnn/distributed/distributed_configs.hpp
    api/ttnn/distributed/distributed_nanobind.hpp
    api/ttnn/distributed/distributed_tensor.hpp
    api/ttnn/distributed/fabric_socket.hpp
    api/ttnn/distributed/host_ccl.hpp
    api/ttnn/distributed/isocket.hpp
    api/ttnn/distributed/mpi_socket.hpp
    api/ttnn/distributed/tensor_topology.hpp
    api/ttnn/distributed/types.hpp
    api/ttnn/events.hpp
    api/ttnn/global_circular_buffer.hpp
    api/ttnn/global_semaphore.hpp
    api/ttnn/graph/capture_program_config.hpp
    api/ttnn/graph/graph_consts.hpp
    api/ttnn/graph/graph_serialization.hpp
    api/ttnn/graph/graph_operation_queries.hpp
    api/ttnn/graph/graph_processor.hpp
    api/ttnn/graph/graph_nanobind.hpp
    api/ttnn/graph/graph_query_op_constraints.hpp
    api/ttnn/graph/graph_query_op_runtime.hpp
    api/ttnn/graph/graph_trace_utils.hpp
    api/ttnn/graph/levelized_graph.hpp
    api/ttnn/mesh_device_operation_adapter.hpp
    api/ttnn/mesh_device_operation_utils.hpp
    api/ttnn/metal_v2_artifacts.hpp
    api/ttnn/operation.hpp
    api/ttnn/operation_concepts.hpp
    api/ttnn/reports.hpp
    api/ttnn/tensor/host_buffer/functions.hpp
    api/ttnn/tensor/layout/alignment.hpp
    api/ttnn/tensor/layout/layout.hpp
    api/ttnn/tensor/layout/page_config.hpp
    api/ttnn/tensor/layout/tensor_layout.hpp
    api/ttnn/tensor/memory_config/memory_config.hpp
    api/ttnn/tensor/serialization.hpp
    api/ttnn/tensor/shape/shape.hpp
    api/ttnn/tensor/storage.hpp
    api/ttnn/tensor/tensor.hpp
    api/ttnn/tensor/tensor_attributes.hpp
    api/ttnn/tensor/tensor_impl.hpp
    api/ttnn/tensor/tensor_ops.hpp
    api/ttnn/tensor/tensor_spec.hpp
    api/ttnn/tensor/tensor_utils.hpp
    api/ttnn/tensor/to_string.hpp
    api/ttnn/tensor/types.hpp
    api/ttnn/tensor/unit_mesh/unit_mesh_utils.hpp
    api/ttnn/tensor/xtensor/conversion_utils.hpp
    api/ttnn/tensor/xtensor/partition.hpp
    api/ttnn/tensor/xtensor/xtensor_all_includes.hpp
    api/ttnn/types.hpp
    api/ttnn/tensor/py_to_tt_tensor.hpp
    cpp/ttnn/operations/copy/typecast/typecast.hpp
    cpp/ttnn/operations/creation/creation.hpp
    cpp/ttnn/operations/functions.hpp
    cpp/ttnn/operations/compute_throttle_utils.hpp
    cpp/ttnn/operations/cb_utils.hpp
    cpp/ttnn/operations/math.hpp
    cpp/ttnn/operations/trace.hpp
)
