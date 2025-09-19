// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_distributed_sdpa_op.hpp"

#include "ring_distributed_sdpa_program_factory.hpp"
#include "ttnn/run_operation.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::transformer {

void RingDistributedScaledDotProductAttention::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 3, "Must have 3 input tensors (Q, K, V) for ring-distributed SDPA");

    const auto& input_tensor_q = input_tensors.at(0);
    const auto& input_tensor_k = input_tensors.at(1);
    const auto& input_tensor_v = input_tensors.at(2);

    // Ring parameter validation
    TT_FATAL(this->ring_size > 0, "ring_size must be greater than 0, got {}", this->ring_size);
    TT_FATAL(this->ring_size <= 32, "ring_size must be <= 32 for practical use, got {}", this->ring_size);

    // Validate ring_id if provided
    if (this->ring_id.has_value()) {
        TT_FATAL(
            this->ring_id.value() < this->ring_size,
            "ring_id must be less than ring_size, got ring_id={}, ring_size={}",
            this->ring_id.value(),
            this->ring_size);
    }

    // Ring distribution requires even number of chunks for load balancing
    TT_FATAL(this->ring_size % 2 == 0, "ring_size must be even for balanced distribution, got {}", this->ring_size);

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
    for (auto& tensor : input_tensors) {
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

    // Ring-distributed SDPA is causal-only
    TT_FATAL(
        Sq == Sk,
        "Ring-distributed SDPA is causal and requires Q and K to have the same sequence length. Got Q: {}, K: {}",
        Sq,
        Sk);

    // Basic Q,K,V shape validation
    TT_FATAL(
        k_shape[0] == B && v_shape[0] == B,
        "Batch sizes must match. Got Q: {}, K: {}, V: {}",
        B,
        k_shape[0],
        v_shape[0]);
    TT_FATAL(v_shape[1] == nkv, "K and V num_heads must match. Got K: {}, V: {}", k_shape[1], v_shape[1]);
    TT_FATAL(v_shape[2] == Sk, "K and V sequence length must match. Got K: {}, V: {}", k_shape[2], v_shape[2]);
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
        Sq / tt::constants::TILE_WIDTH >= 2 * this->ring_size,
        "Sequence length tiles must be at least 2*ring_size for ring distribution. Got seq_len: {}, ring_size: {}",
        Sq,
        this->ring_size);
    TT_FATAL(
        Sq % (2 * this->ring_size) == 0,
        "Sequence length must be divisible by 2*ring_size for even chunk distribution. Got seq_len: {}, ring_size: {}",
        Sq,
        this->ring_size);

    // Chunk size compatibility
    const auto q_chunk_size = this->get_q_chunk_size();
    const auto k_chunk_size = this->get_k_chunk_size();
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
        q_chunk_size < Sq / ring_size,
        "q_chunk_size must be less than sequence length tiles divided by ring size. Got q_chunk_size: {}, sequence "
        "length tiles: {}, ring size: {}",
        q_chunk_size,
        Sq / ring_size,
        ring_size);

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

std::uint32_t RingDistributedScaledDotProductAttention::get_q_chunk_size() const {
    return this->program_config ? this->program_config->q_chunk_size : 32;
}

std::uint32_t RingDistributedScaledDotProductAttention::get_k_chunk_size() const {
    return this->program_config ? this->program_config->k_chunk_size : 32;
}

std::vector<TensorSpec> RingDistributedScaledDotProductAttention::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_q = input_tensors.at(0);
    const auto& q_shape = input_tensor_q.logical_shape();

    // Calculate local output shape: each device processes subset of queries
    const auto global_seq_len = q_shape[2];
    const auto chunk_size = global_seq_len / (2 * this->ring_size);
    const auto local_seq_len = 2 * chunk_size;  // Each device gets 2 chunks

    // Local output shape: [B, NQH, local_seq_len, DH]
    auto local_output_shape = input_tensor_q.logical_shape();
    local_output_shape[2] = local_seq_len;  // Update sequence length dimension

    return {TensorSpec(
        local_output_shape, TensorLayout(input_tensor_q.dtype(), PageConfig(Layout::TILE), output_mem_config))};
}

operation::MeshWorkloadWithCallbacks RingDistributedScaledDotProductAttention::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

operation::ProgramWithCallbacks RingDistributedScaledDotProductAttention::create_program_at(
    const ttnn::MeshCoordinate& coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);

    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1]));
    }

    std::size_t q_chunk_size = this->get_q_chunk_size();
    std::size_t k_chunk_size = this->get_k_chunk_size();

    // Determine ring_id: use provided value or infer from device coordinate
    uint32_t ring_id;
    if (this->ring_id.has_value()) {
        // Use explicitly provided ring_id
        ring_id = this->ring_id.value();
    } else {
        // Infer ring_id from device coordinate (similar to ring_joint_sdpa)
        auto mesh_device = input_tensors[0].device();
        IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();

        // Get all devices in the ring (assuming linear layout along one axis)
        const auto& mesh_view = mesh_device->get_view();
        std::vector<IDevice*> devices_to_use;
        // For simplicity, assume ring is along the first axis (adjust as needed)
        if (mesh_view.shape()[0] == this->ring_size) {
            devices_to_use = mesh_view.get_devices_on_column(coord[1]);
        } else if (mesh_view.shape()[1] == this->ring_size) {
            devices_to_use = mesh_view.get_devices_on_row(coord[0]);
        } else {
            TT_FATAL(
                false,
                "Ring size {} doesn't match mesh dimensions [{}, {}]",
                this->ring_size,
                mesh_view.shape()[0],
                mesh_view.shape()[1]);
        }

        // Find ring_id (device index in the ring)
        ring_id = 0;
        for (uint32_t i = 0; i < this->ring_size; ++i) {
            if (devices_to_use.at(i) == target_device) {
                ring_id = i;
                break;
            }
        }

        log_debug(tt::LogOp, "Inferred ring_id: {} for device_id: {}", ring_id, target_device->id());
    }

    return detail::ring_sdpa_multi_core(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        output_tensor,
        this->ring_size,
        ring_id,
        scale,
        q_chunk_size,
        k_chunk_size,
        this->compute_kernel_config,
        this->program_config);
}

}  // namespace ttnn::operations::transformer
