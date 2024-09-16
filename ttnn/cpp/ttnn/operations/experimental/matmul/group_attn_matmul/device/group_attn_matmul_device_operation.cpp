// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "group_attn_matmul_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"

namespace ttnn::operations::experimental::matmul {

void GroupAttnMatmulDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    // input_a: [q_len, q_heads, batch, head_dim]
    // input_b: [batch, kv_heads, head_dim, kv_len]
    // intermediate: [q_heads, batch, batch, kv_len]
    // output: [q_len, q_heads, batch, kv_len]

    TT_FATAL(input_tensors.size() == 2);
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_FATAL(
        (input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
        "Inputs to matmul must be tilized");

    // TODO: Uplift to support BFLOAT8_B and mixed precision
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to matmul need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");

    const auto ashape = input_tensor_a.get_legacy_shape();
    const auto bshape = input_tensor_b.get_legacy_shape();
    TT_FATAL((ashape[0] == 1), "Input q_len must be 1!");
    TT_FATAL((ashape[1] % bshape[1] == 0), "Number of q_heads must be divisible by kv_heads!");
    TT_FATAL((ashape[2] == bshape[0]), "Num of users must match!");
    TT_FATAL((bshape[0] == 32), "Only batch 32 is supported for group attention matmul!");

    const auto num_cores_used = std::max(ashape[1], tt::constants::TILE_HEIGHT);  // Need at least 32 cores for mcasting KV heads
    TT_FATAL(
        (num_cores_used <= this->compute_with_storage_grid_size.x * this->compute_with_storage_grid_size.y),
        "Compute grid size is too small for group attention matmul! For now, we require at most 1 q_heads per core.");

    // Any sharded memory configs must be HEIGHT_SHARDED and have the same orientation
    ShardOrientation shard_orientation = this->row_major ? ShardOrientation::ROW_MAJOR : ShardOrientation::COL_MAJOR;
    if (input_tensor_a.is_sharded()) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(
            input_tensor_a.shard_spec().value().orientation == shard_orientation,
            "Any sharded memory configs must have the same shard orientation as one another!");
        TT_FATAL(
            input_tensor_a.shard_spec().value().num_cores() == ashape[1],
            "Q heads must be sharded on number of q heads!");
        auto shard_shape = input_tensor_a.shard_spec().value().shape;
        TT_FATAL(shard_shape[0] == ashape[2]);
        TT_FATAL(shard_shape[1] == ashape[3]);
    }
    if (input_tensor_b.is_sharded()) {
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(
            input_tensor_b.shard_spec().value().orientation == shard_orientation,
            "Any sharded memory configs must have the same shard orientation as one another!");
        TT_FATAL(input_tensor_b.shard_spec().value().num_cores() == bshape[0], "KV heads must be sharded on batch!");
        auto shard_shape = input_tensor_b.shard_spec().value().shape;
        TT_FATAL(shard_shape[0] == bshape[1] * bshape[2]);
        TT_FATAL(shard_shape[1] == bshape[3]);
    }
    if (this->output_mem_config.is_sharded()) {
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

        // If user passes in output_mem_config with shard_spec, assert that it is the same as the one calculated in
        // GroupAttnMatmulDeviceOperation::create_output_tensors
        if (this->output_mem_config.shard_spec.has_value()) {
            const tt::tt_metal::LegacyShape output_shape = this->compute_output_shapes(input_tensors).at(0);
            const uint32_t num_cores = output_shape[1];
            CoreRangeSet all_cores =
                num_cores_to_corerange_set(num_cores, this->compute_with_storage_grid_size, this->row_major);

            auto shard_shape = this->output_mem_config.shard_spec.value().shape;
            TT_FATAL(
                this->output_mem_config.shard_spec.value().grid == all_cores,
                "Shard spec in output mem config must match shard spec calculated in "
                "GroupAttnMatmulDeviceOperation::create_output_tensors!");
            TT_FATAL(
                this->output_mem_config.shard_spec.value().orientation == shard_orientation,
                "Any sharded memory configs must have the same shard orientation as one another!");
            TT_FATAL(shard_shape[0] == output_shape[2]);
            TT_FATAL(shard_shape[1] == output_shape[3]);
        }
    }

    bool read_from_kv_cache = false;
    if (this->num_tokens.has_value() or this->transpose_hw.has_value()) {
        TT_FATAL(
            (this->num_tokens.has_value() and this->transpose_hw.has_value()),
            "Must provide num_tokens and transpose_hw flag if we are reading from cache for in1!");
        TT_FATAL(this->num_tokens.value() % 32 == 0, "Number of tokens must be divisble by 32!");
        read_from_kv_cache = true;
    }

    if (read_from_kv_cache) {
        if (this->transpose_hw.value()) {
            TT_FATAL(
                ashape[3] == bshape[3] &&
                "For pre-attention matmul, dimension K for B is in B.shape[3], so A.shape[3] must match B.shape[3]");  // A.K == B.K
        } else {
            TT_FATAL(
                ashape[3] == this->num_tokens &&
                "For post-attention matmul, dimension K (A.shape[3]) is the kv_seq_len in this case and must match the "
                "length of the cache we read");  // A.K == B.K
        }
    } else {
        TT_FATAL(
            ashape[3] == bshape[2] &&
            "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in attn_matmul op");  // A.K == B.K
    }
}

std::vector<tt::tt_metal::LegacyShape> GroupAttnMatmulDeviceOperation::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // input_a: [q_len, q_heads, batch, head_dim]
    // input_b: [batch, kv_heads, head_dim, kv_len]
    // intermediate: [q_heads, batch, batch, kv_len]
    // output: [q_len, q_heads, batch, kv_len]
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto ashape = input_tensor_a.get_legacy_shape();
    const auto bshape = input_tensor_b.get_legacy_shape();

    uint32_t N = bshape[3];
    if (this->transpose_hw.value_or(false)) {
        N = this->num_tokens.value();
    }

    return {tt::tt_metal::LegacyShape{1, ashape[1], ashape[2], N}};
}

std::vector<Tensor> GroupAttnMatmulDeviceOperation::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    if (this->output_mem_config.is_sharded()) {
        auto output_mem_config = this->output_mem_config;
        if (this->output_mem_config.shard_spec.has_value()) {
            output_mem_config.shard_spec = this->output_mem_config.shard_spec.value();
        } else {
            const tt::tt_metal::LegacyShape output_shape = this->compute_output_shapes(input_tensors).at(0);
            const uint32_t num_cores = output_shape[1];
            CoreRangeSet all_cores =
                num_cores_to_corerange_set(num_cores, this->compute_with_storage_grid_size, this->row_major);

            ShardOrientation shard_orientation =
                this->row_major ? ShardOrientation::ROW_MAJOR : ShardOrientation::COL_MAJOR;
            ShardSpec shard_spec = ShardSpec{all_cores, {output_shape[2], output_shape[3]}, shard_orientation};
            output_mem_config.shard_spec = shard_spec;
        }
        return {create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            this->output_dtype,
            Layout::TILE,
            input_tensor_a.device(),
            output_mem_config)};
    } else {
        const auto& input_tensor = input_tensors.at(0);
        return {create_device_tensor(
                compute_output_shapes(input_tensors).at(0),
                this->output_dtype,
                Layout::TILE,
                input_tensor.device(),
                this->output_mem_config)};
    }
}

operation::ProgramWithCallbacks GroupAttnMatmulDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    auto device_compute_with_storage_grid_size = input_tensor_a.device()->compute_with_storage_grid_size();
    TT_ASSERT(
        (this->compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x &&
         this->compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y),
        "Unsupported grid shape");

    return multi_core_group_attn_matmul(
        input_tensor_a,
        input_tensor_b,
        output_tensor,
        this->num_tokens,
        this->transpose_hw,
        this->out_subblock_w,
        this->compute_with_storage_grid_size,
        this->row_major,
        this->compute_kernel_config);
}

const operation::Hash GroupAttnMatmulDeviceOperation::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    TT_ASSERT(std::holds_alternative<DeviceStorage>(input_tensor_a.storage()), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(input_tensor_a.storage()),__FILE__, __LINE__));
    TT_ASSERT(std::holds_alternative<DeviceStorage>(input_tensor_b.storage()), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(input_tensor_b.storage()),__FILE__, __LINE__));

    return operation::hash_operation<GroupAttnMatmulDeviceOperation>(
        this->transpose_hw,
        this->out_subblock_w,
        this->compute_with_storage_grid_size.str(),
        this->output_mem_config.memory_layout,
        this->output_mem_config.buffer_type,
        this->output_dtype,
        this->row_major,
        std::get<DeviceStorage>(input_tensor_a.storage()).memory_config().memory_layout,
        std::get<DeviceStorage>(input_tensor_a.storage()).memory_config().buffer_type,
        input_tensor_a.dtype(),
        std::get<DeviceStorage>(input_tensor_b.storage()).buffer->device()->id(),
        std::get<DeviceStorage>(input_tensor_b.storage()).memory_config().memory_layout,
        std::get<DeviceStorage>(input_tensor_b.storage()).memory_config().buffer_type,
        input_tensor_b.dtype(),
        std::get<DeviceStorage>(input_tensor_b.storage()).buffer->device()->id());
}


}  // namespace ttnn::operations::experimental::transformer
