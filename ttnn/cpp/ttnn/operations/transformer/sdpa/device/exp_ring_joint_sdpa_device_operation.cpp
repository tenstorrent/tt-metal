// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_perf_model.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

void ExpRingJointSDPADeviceOperation::validate_on_program_cache_miss(
    const ExpRingJointSDPAParams& args, const ExpRingJointSDPAInputs& tensor_args) {
    const auto& input_tensor_q = tensor_args.input_q;

    const auto& joint_tensor_q = tensor_args.joint_q;
    const auto& joint_tensor_k = tensor_args.joint_k;
    const auto& joint_tensor_v = tensor_args.joint_v;

    const auto& gathered_input_tensor_k = tensor_args.gathered_k;
    const auto& gathered_input_tensor_v = tensor_args.gathered_v;

    const std::vector<Tensor> sdpa_input_tensors = {
        input_tensor_q,
        gathered_input_tensor_k,
        gathered_input_tensor_v,
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v};

    // Check that SDPA coregrid does not overlap with AllGather coregrid
    TT_FATAL(args.program_config.has_value(), "Program config must be provided");
    if (!args.ccl_worker_cores.has_value()) {
        const auto strategy = args.core_allocation_strategy;
        if (strategy == ttnn::ccl::CoreAllocationStrategy::COL_MAJOR) {
            TT_FATAL(
                args.ccl_core_grid_offset.x >= args.program_config.value().compute_with_storage_grid_size.x,
                "SDPA coregrid overlaps with AllGather coregrid (column-major)");
        } else {
            TT_FATAL(
                args.ccl_core_grid_offset.y >= args.program_config.value().compute_with_storage_grid_size.y,
                "SDPA coregrid overlaps with AllGather coregrid (row-major)");
        }
    }

    // Validate joint strategy is 'rear'
    TT_FATAL(args.joint_strategy == "rear", "Joint strategy must be 'rear'. Got: {}", args.joint_strategy);

    // Validate all tensors have the same dtype
    const auto dtype = input_tensor_q.dtype();
    for (const auto& tensor : sdpa_input_tensors) {
        TT_FATAL(
            tensor.dtype() == dtype,
            "All tensors must have the same dtype. Expected {}, got {}",
            dtype,
            tensor.dtype());
    }

    // Get shapes
    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = gathered_input_tensor_k.logical_shape();
    const auto& v_shape = gathered_input_tensor_v.logical_shape();
    const auto& joint_q_shape = joint_tensor_q.logical_shape();
    const auto& joint_k_shape = joint_tensor_k.logical_shape();
    const auto& joint_v_shape = joint_tensor_v.logical_shape();

    // Validate storage types and buffers
    for (const auto& tensor : sdpa_input_tensors) {
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to Joint SDPA need to be on device");
        TT_FATAL(tensor.buffer() != nullptr, "Operands to Joint SDPA need to be allocated in buffers on device");
        TT_FATAL(tensor.layout() == Layout::TILE, "Inputs to Joint SDPA must be tilized");
        TT_FATAL(
            tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B ||
                tensor.dtype() == DataType::BFLOAT4_B,
            "Inputs to Joint SDPA must be BF16 or BF8 or BF4");
        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to Joint SDPA need to be in DRAM");
    }

    // Validate input shapes match
    const auto B = q_shape[0];
    const auto NQH = q_shape[1];
    const auto NKH = k_shape[1];
    const auto N_local = q_shape[2];
    const auto N_global = k_shape[2];
    const auto L = joint_q_shape[2];
    const auto DH = q_shape[3];

    TT_FATAL(
        k_shape[0] == B && v_shape[0] == B && joint_q_shape[0] == B && joint_k_shape[0] == B && joint_v_shape[0] == B,
        "Batch sizes must match. Got Q: {}, K: {}, V: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
        B,
        k_shape[0],
        v_shape[0],
        joint_q_shape[0],
        joint_k_shape[0],
        joint_v_shape[0]);

    // Validate head dimensions match
    TT_FATAL(
        k_shape[3] == DH && v_shape[3] == DH && joint_q_shape[3] == DH && joint_k_shape[3] == DH &&
            joint_v_shape[3] == DH,
        "Head dimensions must match. Got Q: {}, K: {}, V: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
        DH,
        k_shape[3],
        v_shape[3],
        joint_q_shape[3],
        joint_k_shape[3],
        joint_v_shape[3]);

    TT_FATAL(
        v_shape[1] == NKH && joint_q_shape[1] == NQH && joint_k_shape[1] == NKH && joint_v_shape[1] == NKH,
        "Num heads must match. Got Q: {}, K: {}, V: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
        NQH,
        NKH,
        v_shape[1],
        joint_q_shape[1],
        joint_k_shape[1],
        joint_v_shape[1]);

    TT_FATAL(
        v_shape[2] == N_global,
        "V sequence length must be equal to global sequence length. Got V: {}, global sequence length: {}",
        v_shape[2],
        N_global);

    TT_FATAL(
        N_global == N_local * args.ring_size,
        "Global sequence length must be equal to local sequence length times ring size. Got global sequence length: "
        "{}, local sequence length: {}, ring size: {}",
        N_global,
        N_local,
        args.ring_size);

    TT_FATAL(
        args.logical_n <= N_global,
        "Logical sequence length must be less than or equal to global sequence length. Got logical sequence length: "
        "{}, global sequence length: {}",
        args.logical_n,
        N_global);

    TT_FATAL(
        (N_global - args.logical_n) < N_local,
        "Delta between global (padded) and logical (unpadded) sequence length must be less than local (per device) "
        "sequence length. Got delta: {}, local sequence length: {} "
        "This implies at least one device will have only padded tokens and no real tokens to process. Either "
        "reduce the ring size or reduce padding by reducing the chunk size.",
        N_global - args.logical_n,
        N_local);

    TT_FATAL(
        joint_k_shape[2] == L && joint_v_shape[2] == L,
        "Joint sequence length must match. Got joint_K: {}, joint_V: {}",
        joint_k_shape[2],
        joint_v_shape[2]);

    // Check shapes based on ring
    TT_FATAL(
        q_shape[2] * args.ring_size == k_shape[2],
        "Q sequence length times ring size must be equal to K sequence length. Got Q: {}, K: {}, ring_size: {}",
        q_shape[2],
        k_shape[2],
        args.ring_size);
    TT_FATAL(
        k_shape[2] == v_shape[2],
        "K sequence length must be equal to V sequence length. Got K: {}, V: {}",
        k_shape[2],
        v_shape[2]);

    TT_FATAL(NQH == NKH, "Q num_heads must be equal to K num_heads. Got Q: {}, K: {}", NQH, NKH);

    // Validate chunk sizes if program config is provided
    auto q_chunk_size = args.get_q_chunk_size();
    auto k_chunk_size = args.get_k_chunk_size();

    TT_FATAL(
        q_chunk_size % tt::constants::TILE_WIDTH == 0,
        "q_chunk_size must be divisible by TILE_SIZE. Got q_chunk_size: {}, TILE_SIZE: {}",
        q_chunk_size,
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        k_chunk_size % tt::constants::TILE_WIDTH == 0,
        "k_chunk_size must be divisible by TILE_SIZE. Got k_chunk_size: {}, TILE_SIZE: {}",
        k_chunk_size,
        tt::constants::TILE_WIDTH);

    TT_FATAL(
        N_local % tt::constants::TILE_HEIGHT == 0,
        "Local sequence length must be divisible by TILE_HEIGHT. Got N_local: {}, TILE_HEIGHT: {}",
        N_local,
        tt::constants::TILE_HEIGHT);

    // Validate padding: Only the sequence dimension may be padded
    auto validate_padding = [](const Tensor& tensor) {
        const auto& logical_shape = tensor.logical_shape();
        const auto& padded_shape = tensor.padded_shape();
        TT_FATAL(logical_shape[0] == padded_shape[0], "Padding is not supported on the batch dimension");
        TT_FATAL(logical_shape[1] == padded_shape[1], "Padding is not supported on the num_heads dimension");
        TT_FATAL(logical_shape[3] == padded_shape[3], "Padding is not supported on the head_dim dimension");
    };

    for (const auto& tensor : sdpa_input_tensors) {
        validate_padding(tensor);
    }
}

ExpRingJointSDPAResultSpec ExpRingJointSDPADeviceOperation::compute_output_specs(
    const ExpRingJointSDPAParams& args, const ExpRingJointSDPAInputs& tensor_args) {
    const auto& input = tensor_args.input_q;
    const auto& joint_input = tensor_args.joint_q;
    auto stats_shape = input.logical_shape();
    stats_shape[3] = 1;
    // 2× the sequence length: first half stores running max, second half stores running sum.
    // Used as DRAM scratch for multi-Q-chunk deferred norm round-trips between ring iterations.
    stats_shape[2] = (input.padded_shape()[2] + joint_input.padded_shape()[2]) * 2;

    return {
        TensorSpec(
            input.logical_shape(),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config)),
        TensorSpec(
            joint_input.logical_shape(),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config)),
        TensorSpec(stats_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config))};
}

ExpRingJointSDPAResult ExpRingJointSDPADeviceOperation::create_output_tensors(
    const ExpRingJointSDPAParams& args, const ExpRingJointSDPAInputs& tensor_args) {
    auto output_specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(output_specs[EXP_RING_JOINT_SDPA_OUTPUT_IDX], tensor_args.input_q.device()),
        create_device_tensor(output_specs[EXP_RING_JOINT_SDPA_JOINT_OUTPUT_IDX], tensor_args.joint_q.device()),
        create_device_tensor(output_specs[EXP_RING_JOINT_SDPA_STATS_OUTPUT_IDX], tensor_args.input_q.device()),
    };
}

tt::stl::hash::hash_t ExpRingJointSDPADeviceOperation::compute_program_hash(
    const ExpRingJointSDPAParams& args, const ExpRingJointSDPAInputs& tensor_args) {
    const std::vector<Tensor> input_tensors = {
        tensor_args.input_q,
        tensor_args.input_k,
        tensor_args.input_v,
        tensor_args.joint_q,
        tensor_args.joint_k,
        tensor_args.joint_v,
        tensor_args.gathered_k,
        tensor_args.gathered_v,
    };
    return tt::tt_metal::operation::hash_operation<ExpRingJointSDPADeviceOperation>(
        input_tensors,
        args.joint_strategy,
        args.scale,
        args.logical_n,
        args.ring_size,
        args.compute_kernel_config,
        args.program_config,
        args.dim,
        args.num_links,
        args.cluster_axis,
        args.ccl_core_grid_offset);
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensors>
ExpRingJointSDPADeviceOperation::create_op_performance_model(
    const ExpRingJointSDPAParams& args,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& output_tensors) {
    Tensors input_tensors = {
        tensor_args.input_q,
        tensor_args.input_k,
        tensor_args.input_v,
        tensor_args.joint_q,
        tensor_args.joint_k,
        tensor_args.joint_v,
        tensor_args.gathered_k,
        tensor_args.gathered_v};

    auto& output_tensor = output_tensors[EXP_RING_JOINT_SDPA_OUTPUT_IDX];
    auto arch = output_tensor.storage_type() == StorageType::DEVICE ? output_tensor.device()->arch()
                                                                    : ttnn::GetDefaultDevice()->arch();

    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        log_warning(tt::LogOp, "ExpRingJointSDPA perf model does not support arch '{}'", enchantum::to_string(arch));
        return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, 0);
    }

    const auto& q_shape = tensor_args.input_q.logical_shape();
    const auto& gathered_k_shape = tensor_args.gathered_k.logical_shape();
    const auto& v_shape = tensor_args.gathered_v.logical_shape();
    const auto& joint_q_shape = tensor_args.joint_q.logical_shape();

    CoreCoord grid = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                     : output_tensor.device()->compute_with_storage_grid_size();
    MathFidelity fidelity = ttnn::get_math_fidelity(args.compute_kernel_config);

    const uint32_t B = q_shape[0];
    const uint32_t NQH = q_shape[1];
    const uint32_t N_local = q_shape[2];
    const uint32_t N_global = gathered_k_shape[2];
    const uint32_t L = joint_q_shape[2];
    const uint32_t DH = q_shape[3];
    const uint32_t DV = v_shape[3];

    // ExpRingJointSDPA: local Q and joint Q attend to (gathered K + joint K)
    // Total Q dimension: N_local + L, Total K dimension: N_global + L
    const uint32_t cat_Sq = N_local + L;
    const uint32_t cat_Sk = N_global + L;

    // Single attention pass over concatenated dimensions, non-causal
    int ideal_cycles = operations::transformer::sdpa::compute_sdpa_ideal_cycles(
        B, NQH, cat_Sq, cat_Sk, DH, DV, false, fidelity, grid.x * grid.y);

    return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, ideal_cycles);
}

}  // namespace ttnn::prim

namespace ttnn::prim {

ExpRingJointSDPAResult exp_ring_joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& joint_tensor_q,
    const ttnn::Tensor& joint_tensor_k,
    const ttnn::Tensor& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    const std::string& joint_strategy,
    const std::size_t logical_n,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const CoreCoord ccl_core_grid_offset,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<float> scale,
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    std::optional<std::vector<CoreCoord>> ccl_worker_cores,
    const uint32_t num_workers_per_link,
    const uint32_t num_buffers_per_channel) {
    using OperationType = ttnn::prim::ExpRingJointSDPADeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather invoked with cluster_axis API without 2D mesh, which is currently unsupported");
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    int32_t rank = input_tensor_k.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    auto operation_attributes = OperationType::operation_attributes_t(
        joint_strategy,
        scale,
        logical_n,
        num_devices,
        tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        std::move(program_config),
        kernel_config_val,
        gather_dim,
        num_links,
        topology,
        multi_device_global_semaphore,
        subdevice_id,
        core_allocation_strategy,
        cluster_axis,
        ccl_core_grid_offset,
        std::move(ccl_worker_cores),
        num_workers_per_link,
        num_buffers_per_channel);

    auto tensor_args = OperationType::tensor_args_t{
        .input_q = input_tensor_q,
        .input_k = input_tensor_k,
        .input_v = input_tensor_v,
        .joint_q = joint_tensor_q,
        .joint_k = joint_tensor_k,
        .joint_v = joint_tensor_v,
        .gathered_k = persistent_output_buffer_k,
        .gathered_v = persistent_output_buffer_v};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
