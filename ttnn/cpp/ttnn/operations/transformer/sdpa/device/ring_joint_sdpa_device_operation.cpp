// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace experimental::ccl;

void RingJointSDPADeviceOperation::validate_on_program_cache_miss(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    const auto& input_tensor_q = tensor_args.input_q;
    const auto& gathered_input_tensor_k = tensor_args.gathered_k;
    const auto& gathered_input_tensor_v = tensor_args.gathered_v;

    // Validate joint tensor consistency: if any joint tensor is provided, all must be provided
    const bool has_joint_q = tensor_args.joint_q.has_value();
    const bool has_joint_k = tensor_args.joint_k.has_value();
    const bool has_joint_v = tensor_args.joint_v.has_value();

    TT_FATAL(
        (has_joint_q && has_joint_k && has_joint_v) || (!has_joint_q && !has_joint_k && !has_joint_v),
        "Joint tensors must be provided consistently: either all joint tensors (Q, K, V) must be provided, or none. "
        "Got joint_q: {}, joint_k: {}, joint_v: {}",
        has_joint_q ? "provided" : "missing",
        has_joint_k ? "provided" : "missing",
        has_joint_v ? "provided" : "missing");

    const bool use_joint_tensors = has_joint_q;

    std::vector<Tensor> sdpa_input_tensors = {input_tensor_q, gathered_input_tensor_k, gathered_input_tensor_v};

    // Add joint tensors to validation only if they exist
    if (use_joint_tensors) {
        sdpa_input_tensors.insert(
            sdpa_input_tensors.end(),
            {tensor_args.joint_q.value(), tensor_args.joint_k.value(), tensor_args.joint_v.value()});
    }

    ttnn::experimental::prim::RingAttentionAllGatherAsyncDeviceOperation::validate_on_program_cache_miss(
        args.all_gather_operation_attributes, args.all_gather_tensor_args);

    // Check that SDPA coregrid does not overlap with AllGather coregrid
    TT_FATAL(args.program_config.has_value(), "Program config must be provided");
    const auto strategy = args.all_gather_operation_attributes.core_allocation_strategy;
    if (strategy == ttnn::ccl::CoreAllocationStrategy::COL_MAJOR) {
        TT_FATAL(
            args.ccl_core_grid_offset.x >= args.program_config.value().compute_with_storage_grid_size.x,
            "SDPA coregrid overlaps with AllGather coregrid (column-major)");
    } else {
        TT_FATAL(
            args.ccl_core_grid_offset.y >= args.program_config.value().compute_with_storage_grid_size.y,
            "SDPA coregrid overlaps with AllGather coregrid (row-major)");
    }

    // Validate joint strategy is 'rear' when joint tensors are provided
    if (use_joint_tensors) {
        TT_FATAL(
            !args.joint_strategy.has_value() || args.joint_strategy.value() == "rear",
            "Joint strategy must be 'rear' when provided. Got: {}",
            args.joint_strategy.value_or("none"));
    }

    // Validate all tensors have the same dtype
    const auto dtype = input_tensor_q.dtype();
    if (!args.is_causal) {
        for (const auto& tensor : sdpa_input_tensors) {
            TT_FATAL(
                tensor.dtype() == dtype,
                "All tensors must have the same dtype. Expected {}, got {}",
                dtype,
                tensor.dtype());
        }
    }

    // Get shapes
    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = gathered_input_tensor_k.logical_shape();
    const auto& v_shape = gathered_input_tensor_v.logical_shape();

    // Get joint tensor shapes only if they exist
    ttnn::Shape joint_q_shape, joint_k_shape, joint_v_shape;
    if (use_joint_tensors) {
        joint_q_shape = tensor_args.joint_q.value().logical_shape();
        joint_k_shape = tensor_args.joint_k.value().logical_shape();
        joint_v_shape = tensor_args.joint_v.value().logical_shape();
    }

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
    const auto NVH = v_shape[1];
    const auto N_local = q_shape[2];
    const auto N_global = k_shape[2];
    const auto DH = q_shape[3];

    // Calculate L (joint sequence length) conditionally
    uint32_t L = 0;  // Default joint sequence length
    if (use_joint_tensors) {
        const auto& joint_q_shape = tensor_args.joint_q.value().logical_shape();
        L = joint_q_shape[2];
    }

    auto q_chunk_size = args.get_q_chunk_size();
    auto k_chunk_size = args.get_k_chunk_size();

    TT_FATAL(!(L != 0 && args.is_causal), "Causality is enabled only for ring attention");

    TT_FATAL(
        !(args.is_balanced && (N_local / 2) % q_chunk_size != 0),
        "q_chunk_size must divide half of local q seq_len in balanced case");

    TT_FATAL(
        k_shape[0] == B && v_shape[0] == B,
        "Batch sizes must match. Got Q: {}, K: {}, V: {}",
        B,
        k_shape[0],
        v_shape[0]);

    if (use_joint_tensors) {
        TT_FATAL(
            joint_q_shape[0] == B && joint_k_shape[0] == B && joint_v_shape[0] == B,
            "Joint tensor batch sizes must match input tensors. Got joint_Q: {}, joint_K: {}, joint_V: {}, expected: "
            "{}",
            joint_q_shape[0],
            joint_k_shape[0],
            joint_v_shape[0],
            B);
    }

    // Validate head dimensions match
    if (!args.is_causal) {
        TT_FATAL(
            k_shape[3] == DH && v_shape[3] == DH,
            "Head dimensions must match. Got Q: {}, K: {}, V: {}",
            DH,
            k_shape[3],
            v_shape[3]);

        if (use_joint_tensors) {
            TT_FATAL(
                joint_q_shape[3] == DH && joint_k_shape[3] == DH && joint_v_shape[3] == DH,
                "Joint tensor head dimensions must match. Got joint_Q: {}, joint_K: {}, joint_V: {}, expected: {}",
                joint_q_shape[3],
                joint_k_shape[3],
                joint_v_shape[3],
                DH);
        }
    } else {
        TT_FATAL(k_shape[3] == DH, "Q/K head dimensions must match. Got Q: {}, K: {}", DH, k_shape[3]);
    }

    if (use_joint_tensors) {
        TT_FATAL(
            joint_q_shape[1] == NQH && joint_k_shape[1] == NKH && joint_v_shape[1] == NVH,
            "Joint tensor num heads must match input tensors. Got joint_Q: {}, joint_K: {}, joint_V: {}, expected Q: "
            "{}, K: {}, V: {}",
            joint_q_shape[1],
            joint_k_shape[1],
            joint_v_shape[1],
            NQH,
            NKH,
            NVH);
    }

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

    if (use_joint_tensors) {
        TT_FATAL(
            joint_k_shape[2] == L && joint_v_shape[2] == L,
            "Joint sequence length must match. Got joint_K: {}, joint_V: {}, expected: {}",
            joint_k_shape[2],
            joint_v_shape[2],
            L);
    }

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

    TT_FATAL(NQH == NVH, "Q num_heads must be equal to V num_heads. Got Q: {}, V: {}", NQH, NVH);
    TT_FATAL(NKH == NVH || NKH == 1, "K num_heads must be equal to V num_heads or 1. Got K: {}, V: {}", NKH, NVH);

    // Validate chunk sizes if program config is provided

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

RingJointSDPAResultSpec RingJointSDPADeviceOperation::compute_output_specs(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    const auto& input = tensor_args.input_q;
    auto lse_shape = input.logical_shape();
    lse_shape[3] = 1;

    // Add joint padding to LSE only if joint tensors are provided
    if (tensor_args.joint_q.has_value()) {
        lse_shape[2] = input.padded_shape()[2] + tensor_args.joint_q.value().padded_shape()[2];
    } else {
        lse_shape[2] = input.padded_shape()[2];
    }

    auto out_shape = input.logical_shape();

    // head dim as v head dim
    out_shape[3] = tensor_args.input_v.logical_shape()[3];

    std::optional<TensorSpec> joint_output_spec = std::nullopt;
    if (tensor_args.joint_q.has_value()) {
        joint_output_spec = TensorSpec(
            tensor_args.joint_q.value().logical_shape(),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config));
    }

    return {
        .output = TensorSpec(
            out_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config)),
        .joint_output = joint_output_spec,
        .lse_output = TensorSpec(
            lse_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config))};
}

RingJointSDPAResult RingJointSDPADeviceOperation::create_output_tensors(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    auto output_specs = compute_output_specs(args, tensor_args);

    std::optional<Tensor> joint_output = std::nullopt;
    if (output_specs.joint_output.has_value() && tensor_args.joint_q.has_value()) {
        joint_output = create_device_tensor(output_specs.joint_output.value(), tensor_args.joint_q.value().device());
    }

    return {
        .output = create_device_tensor(output_specs.output, tensor_args.input_q.device()),
        .joint_output = joint_output,
        .lse_output = create_device_tensor(output_specs.lse_output, tensor_args.input_q.device()),
    };
}

tt::stl::hash::hash_t RingJointSDPADeviceOperation::compute_program_hash(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    std::vector<Tensor> input_tensors = {
        tensor_args.input_q,
        tensor_args.input_k,
        tensor_args.input_v,
        tensor_args.gathered_k,
        tensor_args.gathered_v,
    };

    // Add joint tensors to hash only if they exist
    if (tensor_args.joint_q.has_value() && tensor_args.joint_k.has_value() && tensor_args.joint_v.has_value()) {
        input_tensors.insert(
            input_tensors.end(),
            {tensor_args.joint_q.value(), tensor_args.joint_k.value(), tensor_args.joint_v.value()});
    }

    return tt::tt_metal::operation::hash_operation<RingJointSDPADeviceOperation>(
        input_tensors,
        args.joint_strategy.value_or(""),
        args.scale,
        args.is_causal,
        args.is_balanced,
        args.logical_n,
        args.ring_size,
        args.compute_kernel_config,
        args.program_config,
        args.ccl_core_grid_offset,
        ttnn::experimental::prim::RingAttentionAllGatherAsyncDeviceOperation::compute_program_hash(
            args.all_gather_operation_attributes, args.all_gather_tensor_args) /*all_gather input tensors*/
    );
}

}  // namespace ttnn::prim

namespace ttnn::prim {

RingJointSDPAResult ring_joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
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
    const bool is_causal,
    const bool is_balanced,
    const std::optional<float> scale,
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    const std::optional<std::string>& joint_strategy) {
    using OperationType = ttnn::prim::RingJointSDPADeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    /**
     * Create RingAttentionAllGatherAsync struct.
     * It will be a member of the RingJointScaledDotProductAttention struct.
     */
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

    auto all_gather_operation_attributes = ttnn::experimental::prim::RingAttentionAllGatherAsyncParams{
        {},
        gather_dim,
        num_links,
        num_devices,
        input_tensor_k.memory_config(),
        topology,
        multi_device_global_semaphore,
        subdevice_id,
        cluster_axis,
        core_allocation_strategy};
    auto all_gather_tensor_args = ttnn::experimental::prim::RingAttentionAllGatherAsyncInputs{
        {input_tensor_k, input_tensor_v}, {persistent_output_buffer_k, persistent_output_buffer_v}};

    auto operation_attributes = OperationType::operation_attributes_t(
        joint_strategy,
        scale,
        is_causal,
        is_balanced,
        logical_n,
        num_devices,
        tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        std::move(program_config),
        kernel_config_val,
        std::move(all_gather_operation_attributes),
        std::move(all_gather_tensor_args),
        ccl_core_grid_offset);

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
