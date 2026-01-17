// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_distributed_sdpa_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ring_distributed_sdpa_program_factory.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::transformer::ring_distributed_sdpa {

RingDistributedSdpaDeviceOperation::program_factory_t RingDistributedSdpaDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::RingDistributedSdpaMeshWorkloadFactory{};
}

void RingDistributedSdpaDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void RingDistributedSdpaDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    std::vector<Tensor> input_tensors = {tensor_args.q, tensor_args.k, tensor_args.v};

    const auto& input_tensor_q = tensor_args.q;
    const auto& input_tensor_k = tensor_args.k;
    const auto& input_tensor_v = tensor_args.v;

    // Ring parameter validation
    TT_FATAL(
        operation_attributes.ring_size > 0, "ring_size must be greater than 0, got {}", operation_attributes.ring_size);
    TT_FATAL(
        operation_attributes.ring_size <= 32,
        "ring_size must be <= 32 for practical use, got {}",
        operation_attributes.ring_size);

    // Validate ring_id if provided
    if (operation_attributes.ring_id.has_value()) {
        TT_FATAL(
            operation_attributes.ring_id.value() < operation_attributes.ring_size,
            "ring_id must be less than ring_size, got ring_id={}, ring_size={}",
            operation_attributes.ring_id.value(),
            operation_attributes.ring_size);
    }

    // Ring distribution requires even number of chunks for load balancing
    TT_FATAL(
        operation_attributes.ring_size % 2 == 0,
        "ring_size must be even for balanced distribution, got {}",
        operation_attributes.ring_size);

    // Validate all tensors have the same dtype
    const auto dtype = input_tensor_q.dtype();
    for (const auto& tensor : input_tensors) {
        TT_FATAL(
            tensor.dtype() == dtype,
            "All tensors must have the same dtype. Expected {}, got {}",
            dtype,
            tensor.dtype());
    }

    // Validate storage types and buffers
    for (const auto& tensor : input_tensors) {
        TT_FATAL(
            tensor.storage_type() == StorageType::DEVICE, "Operands to ring-distributed SDPA need to be on device");
        TT_FATAL(
            tensor.buffer() != nullptr, "Operands to ring-distributed SDPA need to be allocated in buffers on device");
        TT_FATAL(tensor.layout() == Layout::TILE, "Inputs to ring-distributed SDPA must be tilized");
        TT_FATAL(
            tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B ||
                tensor.dtype() == DataType::BFLOAT4_B,
            "Inputs to ring-distributed SDPA must be BF16, BF8, or BF4");
        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to ring-distributed SDPA need to be in DRAM");
    }

    // Get shapes for validation
    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = input_tensor_k.logical_shape();
    const auto& v_shape = input_tensor_v.logical_shape();
    const auto B = q_shape[0];
    const auto nqh = q_shape[1];
    const auto nkv = k_shape[1];
    const auto Sq = q_shape[2];
    const auto DH = q_shape[3];
    const auto Sk = k_shape[2];
    const auto q_chunk_size =
        operation_attributes.program_config ? operation_attributes.program_config->q_chunk_size : 32;
    const auto k_chunk_size =
        operation_attributes.program_config ? operation_attributes.program_config->k_chunk_size : 32;

    // Validate chunk_start_idx and page_table
    bool is_chunked = operation_attributes.chunk_start_idx.has_value();
    bool has_page_table = tensor_args.page_table.has_value();

    // Validate page_table if provided
    if (has_page_table) {
        const auto& page_table_tensor = tensor_args.page_table.value();
        TT_FATAL(page_table_tensor.storage_type() == StorageType::DEVICE, "page_table tensor must be on device");
        TT_FATAL(page_table_tensor.buffer() != nullptr, "page_table tensor must be allocated in a buffer on device");
        TT_FATAL(
            page_table_tensor.dtype() == DataType::INT32,
            "page_table tensor must have INT32 dtype. Got {}",
            page_table_tensor.dtype());
        const auto& page_table_shape = page_table_tensor.logical_shape();
        TT_FATAL(
            page_table_shape.size() == 2,
            "page_table must be 2D tensor [batch_size x num_pages]. Got shape: {}",
            page_table_shape);
        TT_FATAL(
            page_table_shape[0] == B,
            "page_table batch size must match input batch size. Got page_table batch: {}, input batch: {}",
            page_table_shape[0],
            B);
    }

    if (is_chunked) {
        TT_FATAL(
            has_page_table,
            "page_table must be provided when chunk_start_idx is set. chunk_start_idx: {}",
            operation_attributes.chunk_start_idx.value());
        TT_FATAL(
            operation_attributes.chunk_start_idx.value() >= 0,
            "chunk_start_idx must be non-negative. Got chunk_start_idx: {}",
            operation_attributes.chunk_start_idx.value());

        const auto q_chunk_size =
            operation_attributes.program_config ? operation_attributes.program_config->q_chunk_size : 32;
        TT_FATAL(
            operation_attributes.chunk_start_idx.value() % q_chunk_size == 0,
            "chunk_start_idx must be a multiple of q_chunk_size. Got chunk_start_idx: {}, q_chunk_size: {}",
            operation_attributes.chunk_start_idx.value(),
            q_chunk_size);

        // In chunked mode with paged KV, k_shape[2] represents block_size, not full sequence length
        // The actual KV cache length is determined by the page_table and block_size
        // For ring distributed SDPA with paged KV, we validate that K and V have matching shapes
        // The page_table will map to the appropriate blocks in the paged cache
        const auto block_size = k_shape[2];
        TT_FATAL(
            block_size > 0,
            "block_size (K's sequence dimension in paged mode) must be positive. Got block_size: {}",
            block_size);
        // Note: Full KV cache length validation is done via page_table, not K/V tensor shapes directly

        TT_FATAL(
            Sq % block_size == 0,
            "Sequence length must be a multiple of block_size. Got sequence length: {}, block_size: {}",
            Sq,
            block_size);

        TT_FATAL(
            operation_attributes.chunk_start_idx.value() % block_size == 0,
            "chunk_start_idx must be a multiple of block_size. Got chunk_start_idx: {}, block_size: {}",
            operation_attributes.chunk_start_idx.value(),
            block_size);

        TT_FATAL(
            (Sq + operation_attributes.chunk_start_idx.value()) % k_chunk_size == 0,
            "sequence length + chunk_start_idx must be divisible by k_chunk_size. Got sequence length: {}, "
            "chunk_start_idx: {}, k_chunk_size: {}",
            Sq,
            operation_attributes.chunk_start_idx.value(),
            k_chunk_size);
    }

    if (!is_chunked) {
        // Ring-distributed SDPA is causal-only
        TT_FATAL(
            Sq == Sk,
            "Ring-distributed SDPA is causal and requires Q and K to have the same sequence length when not using "
            "prefix caching. Got Q: {}, K: {}",
            Sq,
            Sk);

        // Basic Q,K,V shape validation
        TT_FATAL(
            k_shape[0] == B && v_shape[0] == B,
            "Batch sizes must match. Got Q: {}, K: {}, V: {}",
            B,
            k_shape[0],
            v_shape[0]);

        TT_FATAL(v_shape[2] == Sk, "K and V sequence length must match. Got K: {}, V: {}", k_shape[2], v_shape[2]);
    } else {
        // In paged KV mode, k_shape[2] and v_shape[2] represent block_size and should match
        TT_FATAL(
            v_shape[2] == k_shape[2],
            "K and V block_size (sequence dimension in paged mode) must match. Got K: {}, V: {}",
            k_shape[2],
            v_shape[2]);
    }
    TT_FATAL(v_shape[1] == nkv, "K and V num_heads must match. Got K: {}, V: {}", k_shape[1], v_shape[1]);
    TT_FATAL(
        k_shape[3] == DH && v_shape[3] == DH,
        "Head dimensions must match. Got Q: {}, K: {}, V: {}",
        DH,
        k_shape[3],
        v_shape[3]);
    TT_FATAL(
        nqh >= nkv && nqh % nkv == 0,
        "Q num_heads must be >= K num_heads and divisible by K num_heads. Got Q: {}, K: {}",
        nqh,
        nkv);

    // Ring-specific sequence length validation
    TT_FATAL(
        Sq / tt::constants::TILE_WIDTH >= 2 * operation_attributes.ring_size,
        "Sequence length tiles must be at least 2*ring_size for ring distribution. Got seq_len: {}, ring_size: {}",
        Sq,
        operation_attributes.ring_size);
    TT_FATAL(
        Sq % (2 * operation_attributes.ring_size) == 0,
        "Sequence length must be divisible by 2*ring_size for even chunk distribution. Got seq_len: {}, ring_size: {}",
        Sq,
        operation_attributes.ring_size);

    // Chunk size compatibility
    TT_FATAL(
        q_chunk_size % tt::constants::TILE_WIDTH == 0,
        "q_chunk_size must be divisible by TILE_WIDTH. Got q_chunk_size: {}, TILE_WIDTH: {}",
        q_chunk_size,
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        k_chunk_size % tt::constants::TILE_WIDTH == 0,
        "k_chunk_size must be divisible by TILE_WIDTH. Got k_chunk_size: {}, TILE_WIDTH: {}",
        k_chunk_size,
        tt::constants::TILE_WIDTH);

    TT_FATAL(
        q_chunk_size <= Sq / (2 * operation_attributes.ring_size),
        "q_chunk_size must be less than or equal to per-device sequence length. Got q_chunk_size: {}, per-device "
        "sequence length: {}, global sequence length: {}, ring size: {}",
        q_chunk_size,
        Sq / (2 * operation_attributes.ring_size),
        Sq,
        operation_attributes.ring_size);

    TT_FATAL(
        (Sq / (2 * operation_attributes.ring_size)) % q_chunk_size == 0,
        "per-device sequence length must be divisible by q_chunk_size. Got per-device sequence length: {}, "
        "q_chunk_size: {}",
        Sq / (2 * operation_attributes.ring_size),
        q_chunk_size);

    // Validate padding: Only the sequence dimension may be padded
    auto validate_padding = [](const Tensor& tensor) {
        auto logical_shape = tensor.logical_shape();
        auto padded_shape = tensor.padded_shape();
        TT_FATAL(logical_shape[0] == padded_shape[0], "Padding is not supported on the batch dimension");
        TT_FATAL(logical_shape[1] == padded_shape[1], "Padding is not supported on the num_heads dimension");
        TT_FATAL(logical_shape[3] == padded_shape[3], "Padding is not supported on the head_dim dimension");
    };

    for (const auto& tensor : input_tensors) {
        validate_padding(tensor);
    }
}

RingDistributedSdpaDeviceOperation::spec_return_value_t RingDistributedSdpaDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_q = tensor_args.q;
    const auto& q_shape = input_tensor_q.logical_shape();

    // Calculate local output shape: each device processes subset of queries
    const auto global_seq_len = q_shape[2];
    const auto chunk_size = global_seq_len / (2 * operation_attributes.ring_size);
    const auto local_seq_len = 2 * chunk_size;  // Each device gets 2 chunks

    // Local output shape: [B, NQH, local_seq_len, DH]
    auto local_output_shape = q_shape;
    local_output_shape[2] = local_seq_len;  // Update sequence length dimension

    return TensorSpec(
        local_output_shape,
        TensorLayout(input_tensor_q.dtype(), PageConfig(Layout::TILE), operation_attributes.output_mem_config));
}

RingDistributedSdpaDeviceOperation::tensor_return_value_t RingDistributedSdpaDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.q.device());
}

}  // namespace ttnn::operations::transformer::ring_distributed_sdpa

namespace ttnn::prim {

ttnn::operations::transformer::ring_distributed_sdpa::RingDistributedSdpaDeviceOperation::tensor_return_value_t
ring_distributed_sdpa(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    uint32_t ring_size,
    std::optional<uint32_t> ring_id,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<ttnn::Tensor>& page_table,
    std::optional<int64_t> chunk_start_idx) {
    using OperationType =
        ttnn::operations::transformer::ring_distributed_sdpa::RingDistributedSdpaDeviceOperation;

    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1]));
    }
    auto operation_attributes = OperationType::operation_attributes_t{
        .ring_size = ring_size,
        .ring_id = ring_id,
        .scale = scale,
        .output_mem_config = output_mem_config,
        .program_config = program_config,
        .compute_kernel_config = compute_kernel_config,
        .chunk_start_idx = chunk_start_idx,
    };

    auto tensor_args = OperationType::tensor_args_t{
        .q = input_tensor_q,
        .k = input_tensor_k,
        .v = input_tensor_v,
        .page_table = page_table,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
