// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/tt_dnn/op_library/transformer_tms/transformer_tms.hpp"

#include "ttnn/experimental/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace operations {
namespace primary {
namespace transformers {

void SplitFusedQKVAndSplitHeads::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.get_legacy_shape()[0];
    // TODO: See issue #1744
    TT_FATAL((input_tensor.get_legacy_shape() == Shape({batch_size, 1, 384, 3072})), "Unsupported input shape");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");

    if (input_tensor.is_sharded() == false) {
        TT_FATAL(batch_size >= 7 && batch_size <= 9, "Input batch size must be between 2 to 9 for bert large TM ops!");
    } else {
        auto bbox = input_tensor.shard_spec().value().grid.bounding_box();
        TT_FATAL(
            (bbox.end_coord.x < this->compute_with_storage_grid_size.x &&
             bbox.end_coord.y < this->compute_with_storage_grid_size.y));
        TT_FATAL(input_tensor.shard_spec().value().grid.ranges().size() == 1);
        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
    }
}

std::vector<Shape> SplitFusedQKVAndSplitHeads::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.get_legacy_shape()[0];
    uint32_t num_heads = this->num_heads;
    uint32_t num_output_tensors = 3;
    uint32_t M = input_tensor.get_legacy_shape()[2];                                    // 384
    uint32_t K = input_tensor.get_legacy_shape()[-1] / num_output_tensors / num_heads;  // 64
    return {
        Shape{batch_size, this->num_heads, M, K},
        Shape{batch_size, this->num_heads, K, M},
        Shape{batch_size, this->num_heads, M, K}};
}

std::vector<Tensor> SplitFusedQKVAndSplitHeads::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    if (input_tensor.is_sharded()) {
        // tensor dim
        uint32_t batch = input_tensor.get_legacy_shape()[0];  // 12
        uint32_t num_heads = this->num_heads;
        uint32_t num_output_tensors = 3;
        uint32_t M = input_tensor.get_legacy_shape()[2];                                    // 384
        uint32_t K = input_tensor.get_legacy_shape()[-1] / num_output_tensors / num_heads;  // 64
        // core range
        CoreRangeSet all_cores = input_tensor.shard_spec().value().grid;
        ShardOrientation shard_orientation = input_tensor.shard_spec().value().orientation;
        auto bbox = all_cores.bounding_box();
        uint32_t num_M_cores = shard_orientation == ShardOrientation::ROW_MAJOR ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;
        // shard spec
        uint32_t per_core_M_qv = (num_heads / num_M_cores) * M;  // 768
        uint32_t per_core_N_qv = K;                              // 64
        ShardSpec shard_spec_qv = ShardSpec{all_cores, {per_core_M_qv, per_core_N_qv}, shard_orientation};
        uint32_t per_core_M_k = (num_heads / num_M_cores) * K;  // 128
        uint32_t per_core_N_k = M;                              // 384
        ShardSpec shard_spec_k = ShardSpec{all_cores, {per_core_M_k, per_core_N_k}, shard_orientation};
        // create sharded tensors
        auto mem_config_qv = this->output_mem_config;
        mem_config_qv.shard_spec = shard_spec_qv;
        auto mem_config_k = this->output_mem_config;
        mem_config_k.shard_spec = shard_spec_k;
        auto out_tensor_q = create_device_tensor(
            Shape{batch, num_heads, M, K},
            input_tensor.get_dtype(),
            Layout::TILE,
            input_tensor.device(),
            mem_config_qv);
        auto out_tensor_k = create_device_tensor(
            Shape{batch, num_heads, K, M}, input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), mem_config_k);
        auto out_tensor_v = create_device_tensor(
            Shape{batch, num_heads, M, K},
            input_tensor.get_dtype(),
            Layout::TILE,
            input_tensor.device(),
            mem_config_qv);
        return {out_tensor_q, out_tensor_k, out_tensor_v};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks SplitFusedQKVAndSplitHeads::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto device_compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    TT_ASSERT(
        (this->compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x &&
         this->compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y),
        "Unsupported grid shape");

    if (input_tensor.is_sharded()) {
        return multi_core_split_query_key_value_and_split_heads_sharded(
            input_tensor, output_tensors, this->compute_with_storage_grid_size);
    } else {
        return multi_core_split_query_key_value_and_split_heads(
            input_tensor, output_tensors, this->compute_with_storage_grid_size);
    }
}

void AttnMatmul::validate(const std::vector<Tensor>& input_tensors) const {
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
    TT_FATAL((bshape[1] == 1), "Number of kv_heads must be 1!");  // TODO: May need to uplift to support falcon-40B
    TT_FATAL((ashape[2] == bshape[0]), "Num of users must match!");

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

std::vector<Shape> AttnMatmul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
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

    return {Shape{1, ashape[1], ashape[2], N}};
}

std::vector<Tensor> AttnMatmul::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks AttnMatmul::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    auto device_compute_with_storage_grid_size = input_tensor_a.device()->compute_with_storage_grid_size();
    TT_ASSERT(
        (this->compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x &&
         this->compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y),
        "Unsupported grid shape");

    return multi_core_attn_matmul(
        input_tensor_a,
        input_tensor_b,
        output_tensor,
        this->num_tokens,
        this->transpose_hw,
        this->compute_with_storage_grid_size,
        this->compute_kernel_config);
}

const operation::Hash AttnMatmul::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    TT_ASSERT(std::holds_alternative<DeviceStorage>(input_tensors.at(0).storage()), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(input_tensors.at(0).get_storage()),__FILE__, __LINE__));
    TT_ASSERT(std::holds_alternative<DeviceStorage>(input_tensors.at(1).storage()), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(input_tensors.at(1).get_storage()),__FILE__, __LINE__));
    return operation::hash_operation<AttnMatmul>(
        this->transpose_hw,
        this->output_mem_config,
        this->output_dtype,
        std::get<DeviceStorage>(input_tensors.at(0).storage()).memory_config(),
        input_tensors.at(0).dtype(),
        std::get<DeviceStorage>(input_tensors.at(1).storage()).memory_config(),
        input_tensors.at(1).dtype());
}

void GroupAttnMatmul::validate(const std::vector<Tensor>& input_tensors) const {
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

    const auto num_cores_used = std::max(ashape[1], TILE_HEIGHT);  // Need at least 32 cores for mcasting KV heads
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
        // GroupAttnMatmul::create_output_tensors
        if (this->output_mem_config.shard_spec.has_value()) {
            const Shape output_shape = this->compute_output_shapes(input_tensors).at(0);
            const uint32_t num_cores = output_shape[1];
            CoreRangeSet all_cores =
                num_cores_to_corerange_set(num_cores, this->compute_with_storage_grid_size, this->row_major);

            auto shard_shape = this->output_mem_config.shard_spec.value().shape;
            TT_FATAL(
                this->output_mem_config.shard_spec.value().grid == all_cores,
                "Shard spec in output mem config must match shard spec calculated in "
                "GroupAttnMatmul::create_output_tensors!");
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

std::vector<Shape> GroupAttnMatmul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
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

    return {Shape{1, ashape[1], ashape[2], N}};
}

std::vector<Tensor> GroupAttnMatmul::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    if (this->output_mem_config.is_sharded()) {
        auto output_mem_config = this->output_mem_config;
        if (this->output_mem_config.shard_spec.has_value()) {
            output_mem_config.shard_spec = this->output_mem_config.shard_spec.value();
        } else {
            const Shape output_shape = this->compute_output_shapes(input_tensors).at(0);
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
        return operation::generic_create_output_tensors(
            *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks GroupAttnMatmul::create_program(
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

const operation::Hash GroupAttnMatmul::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    TT_ASSERT(std::holds_alternative<DeviceStorage>(input_tensor_a.storage()), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(input_tensor_a.storage()),__FILE__, __LINE__));
    TT_ASSERT(std::holds_alternative<DeviceStorage>(input_tensor_b.storage()), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(input_tensor_b.storage()),__FILE__, __LINE__));

    return operation::hash_operation<GroupAttnMatmul>(
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

// SSM eltwise mul
void SSMEltwiseMul::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 2);
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_FATAL(
        (input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
        "Inputs to ssm_eltwise_mul must be tilized");

    // TODO: Uplift to support BFLOAT8_B and mixed precision
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to ssm_eltwise_mul need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to ssm_eltwise_mul need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.device() == input_tensor_b.device(),
        "Operands to ssm_eltwise_mul need to be on the same device!");

    TT_FATAL(
        input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout for input a!");
    TT_FATAL(
        input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout for input b!");
    TT_FATAL(
        input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for input a!");
    TT_FATAL(
        input_tensor_b.get_dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor_b.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for input b!");

    TT_FATAL(
        this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout for output!");
    TT_FATAL(
        this->output_dtype == tt::tt_metal::DataType::BFLOAT16 ||
            this->output_dtype == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for output!");

    const auto ashape = input_tensor_a.get_legacy_shape();
    const auto bshape = input_tensor_b.get_legacy_shape();
    TT_FATAL((ashape[0] == 1 and ashape[1] == 1), "Batch not supported for input a!");
    TT_FATAL((bshape[0] == 1 and bshape[1] == 1), "Batch not supported for input b!");
    TT_FATAL((ashape[2] % TILE_HEIGHT == 0), "Num of users must be multiple of 32 for input a!");
    TT_FATAL((bshape[2] % TILE_HEIGHT == 0), "Num of users must be multiple of 32 for input b!");
    TT_FATAL((ashape[2] == bshape[2]), "Num of users must match in both of the input!");
    TT_FATAL((ashape[3] != bshape[3]), "Use eltwise mul for same size inputs!");
    TT_FATAL(
        (ashape[3] == TILE_WIDTH || ashape[3] == TILE_WIDTH * HIDDEN_SIZE), "Input a width must be 32 or 32*5120!");
    TT_FATAL(
        (bshape[3] == HIDDEN_SIZE || bshape[3] == TILE_WIDTH * HIDDEN_SIZE), "Input b width must be 32 or 32*5120!");
}

std::vector<Shape> SSMEltwiseMul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto shape_a = input_tensor_a.get_legacy_shape();
    const auto shape_b = input_tensor_b.get_legacy_shape();

    return {Shape{shape_a[0], shape_a[1], shape_a[2], TILE_WIDTH * HIDDEN_SIZE}};
}

std::vector<Tensor> SSMEltwiseMul::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks SSMEltwiseMul::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);
    const auto hidden_size = HIDDEN_SIZE;

    auto device_compute_with_storage_grid_size = input_tensor_a.device()->compute_with_storage_grid_size();

    return multi_core_ssm_eltwise_mul(
        input_tensor_a, input_tensor_b, output_tensor, hidden_size, this->math_fidelity, device_compute_with_storage_grid_size);
}


void SSM1DSumReduce::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1);
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL((input_tensor_a.get_layout() == Layout::TILE), "Inputs to ssm_1d_sum_reduce must be tilized");

    // TODO: Uplift to support mixed precision
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to ssm_1d_sum_reduce need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr, "Operands to ssm_1d_sum_reduce need to be allocated in buffers on device!");

    TT_FATAL(
        input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout for input a!");
    TT_FATAL(
        input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for input a!");

    TT_FATAL(
        this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout for output!");
    TT_FATAL(
        this->output_dtype == tt::tt_metal::DataType::BFLOAT16 ||
            this->output_dtype == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for output!");

    constexpr uint32_t latent = 32;
    const auto ashape = input_tensor_a.get_legacy_shape();
    TT_FATAL((ashape[0] == 1 and ashape[1] == 1), "Dim 1 and 2 are expected to be 1 in input a!");
    TT_FATAL((ashape[2] % TILE_HEIGHT == 0), "Batch size must be divisible by 32 for input a!");
    TT_FATAL((ashape[3] % TILE_WIDTH == 0), "Final dim must be a multiple of 32!");
    TT_FATAL(((ashape[3] / TILE_WIDTH) % latent == 0), "Final dim/TILE_SIZE must be a multiple of latent size!");
}

std::vector<Shape> SSM1DSumReduce::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    constexpr uint32_t latent = 32;
    const auto& input_tensor_a = input_tensors.at(0);
    const auto shape_a = input_tensor_a.get_legacy_shape();
    return {Shape{shape_a[0], shape_a[1], shape_a[2], shape_a[3] / latent}};
}

std::vector<Tensor> SSM1DSumReduce::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks SSM1DSumReduce::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    auto device_compute_with_storage_grid_size = input_tensor_a.device()->compute_with_storage_grid_size();
    return multi_core_ssm_1d_sum_reduce(input_tensor_a, output_tensor, math_fidelity, device_compute_with_storage_grid_size);
}

void SSMPrefixScan::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 3, "Expected 3 input tensors (A, Bx, H)");

    const auto& a = input_tensors.at(0);
    const auto& bx = input_tensors.at(1);
    TT_FATAL(a.dtype() == bx.dtype(), "Expected input tensors to have the same data type");
    TT_FATAL(a.layout() == Layout::TILE && bx.layout() == Layout::TILE, "Expected input tensors to be tile layout");
    TT_FATAL(a.get_legacy_shape() == bx.get_legacy_shape(), "Expected input tensors to have the same shape");

    const auto& shape = a.get_legacy_shape();
    TT_FATAL(shape.rank() == 4, "Expected input tensors to be rank 4");
    TT_FATAL(shape[0] == 1 && shape[1] == 1, "Dimension 0 and 1 should be size 1");
    TT_FATAL(shape[2] >= TILE_HEIGHT && shape[2] % TILE_HEIGHT == 0, "Sequence length should be a multiple of 32");

    const auto& h = input_tensors.at(2);
    TT_FATAL(h.dtype() == DataType::BFLOAT16, "Expected initial hidden state to be bfloat16");
    TT_FATAL(h.layout() == Layout::ROW_MAJOR, "Expected initial hidden state to be row-major");
    //TT_FATAL(h.get_legacy_shape() == {1, 1, 1, shape[3]}, "Expected initial hidden state to have the same hidden size as A and Bx");

    TT_FATAL(a.is_sharded() && bx.is_sharded() && h.is_sharded(), "Expected input tensors to be sharded");
    TT_FATAL(a.shard_spec().has_value() && bx.shard_spec().has_value() && h.shard_spec().has_value(), "Expected input tensors to be sharded");
    TT_FATAL(
        a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "Expected A tensor to be row major orientation");
    TT_FATAL(
        bx.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "Expected Bx tensor to be row major orientation");
    TT_FATAL(
        h.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "Expected h tensor to be row major orientation");
}

std::vector<Shape> SSMPrefixScan::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& a = input_tensors.at(0);
    return {a.get_legacy_shape()};
}

std::vector<Tensor> SSMPrefixScan::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks SSMPrefixScan::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    const auto& bx = input_tensors.at(1);
    const auto& h = input_tensors.at(2);
    auto& output = output_tensors.at(0);
    auto device_compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    return multi_core_ssm_prefix_scan(a, bx, h, output, math_fidelity, device_compute_with_storage_grid_size);
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
