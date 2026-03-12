// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_profile_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_profile_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_profile_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_perf_model.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

void RingJointSDPAProfileDeviceOperation::validate_on_program_cache_miss(
    const RingJointSDPAProfileParams& args, const RingJointSDPAProfileInputs& tensor_args) {
    const auto& input_tensor_q = tensor_args.input_q;
    const auto& gathered_input_tensor_k = tensor_args.gathered_k;
    const auto& gathered_input_tensor_v = tensor_args.gathered_v;

    // Validate ring_index is within bounds
    TT_FATAL(
        args.ring_index < args.ring_size,
        "ring_index ({}) must be less than ring_size ({})",
        args.ring_index,
        args.ring_size);

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

    TT_FATAL(args.program_config.has_value(), "Program config must be provided");

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
    uint32_t L = 0;
    if (use_joint_tensors) {
        joint_q_shape = tensor_args.joint_q.value().logical_shape();
        joint_k_shape = tensor_args.joint_k.value().logical_shape();
        joint_v_shape = tensor_args.joint_v.value().logical_shape();
        L = joint_q_shape[2];
    }

    // Validate storage types and buffers
    for (const auto& tensor : sdpa_input_tensors) {
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to Profile SDPA need to be on device");
        TT_FATAL(tensor.buffer() != nullptr, "Operands to Profile SDPA need to be allocated in buffers on device");
        TT_FATAL(tensor.layout() == Layout::TILE, "Inputs to Profile SDPA must be tilized");
        TT_FATAL(
            tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B ||
                tensor.dtype() == DataType::BFLOAT4_B,
            "Inputs to Profile SDPA must be BF16 or BF8 or BF4");
        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to Profile SDPA need to be in DRAM");
    }

    // Validate input shapes match
    const auto B = q_shape[0];
    const auto NQH = q_shape[1];
    const auto NKH = k_shape[1];
    const auto NVH = v_shape[1];
    const auto local_padded_N = q_shape[2];
    const auto N_global = k_shape[2];
    const auto DH = q_shape[3];

    auto q_chunk_size = args.get_q_chunk_size();
    auto k_chunk_size = args.get_k_chunk_size();

    TT_FATAL(!(L != 0 && args.is_causal), "Causality is enabled only for ring attention");

    TT_FATAL(
        !(args.is_balanced && (local_padded_N / 2) % q_chunk_size != 0),
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
        N_global == local_padded_N * args.ring_size,
        "Global sequence length must be equal to local sequence length times ring size. Got global sequence length: "
        "{}, local sequence length: {}, ring size: {}",
        N_global,
        local_padded_N,
        args.ring_size);

    TT_FATAL(
        args.logical_n <= N_global,
        "Logical sequence length must be less than or equal to global sequence length. Got logical sequence length: "
        "{}, global sequence length: {}",
        args.logical_n,
        N_global);

    TT_FATAL(
        (N_global - args.logical_n) < local_padded_N,
        "Delta between global (padded) and logical (unpadded) sequence length must be less than local (per device) "
        "sequence length. Got delta: {}, local sequence length: {} "
        "This implies at least one device will have only padded tokens and no real tokens to process. Either "
        "reduce the ring size or reduce padding by reducing the chunk size.",
        N_global - args.logical_n,
        local_padded_N);

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
        local_padded_N % tt::constants::TILE_HEIGHT == 0,
        "Local sequence length must be divisible by TILE_HEIGHT. Got local_padded_N: {}, TILE_HEIGHT: {}",
        local_padded_N,
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

RingJointSDPAProfileResultSpec RingJointSDPAProfileDeviceOperation::compute_output_specs(
    const RingJointSDPAProfileParams& args, const RingJointSDPAProfileInputs& tensor_args) {
    const auto& input = tensor_args.input_q;
    auto stats_shape = input.logical_shape();
    stats_shape[3] = 1;

    // Add joint padding to stats only if joint tensors are provided
    if (tensor_args.joint_q.has_value()) {
        stats_shape[2] = (input.padded_shape()[2] + tensor_args.joint_q.value().padded_shape()[2]) * 2;
    } else {
        stats_shape[2] = input.padded_shape()[2] * 2;
    }

    auto joint_shape = tensor_args.joint_q.has_value()
                           ? tensor_args.joint_q.value().logical_shape()
                           : ttnn::Shape{1, 1, 32, 32};  // Dummy shape for placeholder tensor when no joint input

    return {
        TensorSpec(
            input.logical_shape(),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config)),
        TensorSpec(joint_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config)),
        TensorSpec(stats_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config))};
}

RingJointSDPAProfileResult RingJointSDPAProfileDeviceOperation::create_output_tensors(
    const RingJointSDPAProfileParams& args, const RingJointSDPAProfileInputs& tensor_args) {
    auto output_specs = compute_output_specs(args, tensor_args);

    Tensor joint_output;
    if (tensor_args.joint_q.has_value()) {
        joint_output =
            create_device_tensor(output_specs[PROFILE_JOINT_OUTPUT_IDX], tensor_args.joint_q.value().device());
    } else {
        // Create minimal dummy tensor for placeholder
        joint_output = create_device_tensor(output_specs[PROFILE_JOINT_OUTPUT_IDX], tensor_args.input_q.device());
    }

    return {
        create_device_tensor(output_specs[PROFILE_OUTPUT_IDX], tensor_args.input_q.device()),
        joint_output,
        create_device_tensor(output_specs[PROFILE_STATS_OUTPUT_IDX], tensor_args.input_q.device()),
    };
}

tt::stl::hash::hash_t RingJointSDPAProfileDeviceOperation::compute_program_hash(
    const RingJointSDPAProfileParams& args, const RingJointSDPAProfileInputs& tensor_args) {
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

    return tt::tt_metal::operation::hash_operation<RingJointSDPAProfileDeviceOperation>(
        input_tensors,
        args.joint_strategy.value_or(""),
        args.scale,
        args.is_causal,
        args.is_balanced,
        args.logical_n,
        args.ring_size,
        args.ring_index,
        args.compute_kernel_config,
        args.program_config);
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensors>
RingJointSDPAProfileDeviceOperation::create_op_performance_model(
    const RingJointSDPAProfileParams& args,
    const RingJointSDPAProfileInputs& tensor_args,
    RingJointSDPAProfileResult& output_tensors) {
    // Conditionally pack joint tensors only if present
    Tensors input_tensors = {
        tensor_args.input_q, tensor_args.input_k, tensor_args.input_v, tensor_args.gathered_k, tensor_args.gathered_v};
    if (tensor_args.joint_q.has_value()) {
        input_tensors.push_back(tensor_args.joint_q.value());
        input_tensors.push_back(tensor_args.joint_k.value());
        input_tensors.push_back(tensor_args.joint_v.value());
    }

    auto& output_tensor = output_tensors[PROFILE_OUTPUT_IDX];
    auto arch = output_tensor.storage_type() == StorageType::DEVICE ? output_tensor.device()->arch()
                                                                    : ttnn::GetDefaultDevice()->arch();

    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        log_warning(
            tt::LogOp, "RingJointSDPAProfile perf model does not support arch '{}'", enchantum::to_string(arch));
        return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, 0);
    }

    const auto& q_shape = tensor_args.input_q.logical_shape();
    const auto& gathered_k_shape = tensor_args.gathered_k.logical_shape();
    const auto& v_shape = tensor_args.gathered_v.logical_shape();

    CoreCoord grid = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                     : output_tensor.device()->compute_with_storage_grid_size();
    MathFidelity fidelity = ttnn::get_math_fidelity(args.compute_kernel_config);

    const uint32_t B = q_shape[0];
    const uint32_t NQH = q_shape[1];
    const uint32_t N_local = q_shape[2];
    const uint32_t N_global = gathered_k_shape[2];
    const uint32_t DH = q_shape[3];
    const uint32_t DV = v_shape[3];

    // Calculate L (joint sequence length) if joint tensors are provided
    uint32_t L = 0;
    if (tensor_args.joint_q.has_value()) {
        L = tensor_args.joint_q.value().logical_shape()[2];
    }

    // RingJointSDPAProfile: local Q and joint Q attend to (gathered K + joint K)
    // Total Q dimension: N_local + L, Total K dimension: N_global + L
    const uint32_t cat_Sq = N_local + L;
    const uint32_t cat_Sk = N_global + L;

    // Single attention pass over concatenated dimensions, non-causal when joints present
    bool is_causal = args.is_causal && (L == 0);
    int ideal_cycles = operations::transformer::sdpa::compute_sdpa_ideal_cycles(
        B, NQH, cat_Sq, cat_Sk, DH, DV, is_causal, fidelity, grid.x * grid.y);

    return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, ideal_cycles);
}

}  // namespace ttnn::prim

namespace ttnn::prim {

RingJointSDPAProfileResult ring_joint_scaled_dot_product_attention_profile(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& gathered_k,
    const ttnn::Tensor& gathered_v,
    const std::size_t ring_size,
    const std::size_t ring_index,
    const std::size_t logical_n,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    const bool is_causal,
    const bool is_balanced,
    const std::optional<float> scale,
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    const std::optional<std::string>& joint_strategy) {
    using OperationType = ttnn::prim::RingJointSDPAProfileDeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    auto operation_attributes = RingJointSDPAProfileParams{
        .joint_strategy = joint_strategy,
        .scale = scale,
        .is_causal = is_causal,
        .is_balanced = is_balanced,
        .logical_n = logical_n,
        .ring_size = ring_size,
        .ring_index = ring_index,
        .output_memory_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        .program_config = std::move(program_config),
        .compute_kernel_config = kernel_config_val,
    };

    auto tensor_args = RingJointSDPAProfileInputs{
        .input_q = input_tensor_q,
        .input_k = input_tensor_k,
        .input_v = input_tensor_v,
        .gathered_k = gathered_k,
        .gathered_v = gathered_v,
        .joint_q = joint_tensor_q,
        .joint_k = joint_tensor_k,
        .joint_v = joint_tensor_v,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
