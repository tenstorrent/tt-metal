// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

// Hard-coded for Falcon7B
void NlpCreateHeadsFalcon7B::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE);

    TT_FATAL(input_shape[2] % TILE_HEIGHT == 0);
    TT_FATAL((input_shape == Shape({input_shape[0], 1, input_shape[2], 4672})), "Unsupported input shape");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
}

std::vector<Shape> NlpCreateHeadsFalcon7B::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    std::vector<Shape> output_shape_vec;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    output_shape_vec = {(Shape) {input_shape[0], 71, input_shape[2], 64}, (Shape) {input_shape[0], 1, input_shape[2], 64}, (Shape) {input_shape[0], 1, input_shape[2], 64}};
    return output_shape_vec;
}

std::vector<Tensor> NlpCreateHeadsFalcon7B::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        TT_ASSERT(false);
        return {};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks NlpCreateHeadsFalcon7B::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    return  multi_core_nlp_create_qkv_heads_falcon7b(input_tensor, output_tensors, compute_with_storage_grid_size);
}

// Generic NLP CreateHeads op for decode
void NlpCreateHeadsDecode::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    // TODO: Rewrite validation for this decode case
    // NOTE: Checks for head_dim and shape[3] is done in nlp_create_qkv_heads because it's needed to infer head_dim
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16, "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE);

    // input
    TT_FATAL(input_shape[3] % TILE_WIDTH == 0, "Unsupported input shape");  // head_dim must be multiple of TILE_WIDTH
    TT_FATAL(input_shape[2] == 32, "Unsupported input shape");  // 32 users
    TT_FATAL(input_shape[1] == 1, "Unsupported input shape");
    TT_FATAL(input_shape[0] == 1, "Unsupported input shape");
    TT_FATAL(input_tensor.is_sharded(), "Input must be sharded");
    TT_FATAL(input_tensor.shard_spec().value().shape[0] == input_tensor.volume() / input_tensor.get_legacy_shape()[-1]);
    TT_FATAL(input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
    // we either put everything in one shard or split it into minimum tile width accross as many cores as possible
    TT_FATAL(input_tensor.shard_spec().value().shape[1] == (this->num_q_heads + this->num_kv_heads * 2) * this->head_dim || input_tensor.shard_spec().value().shape[1] == 32);
    auto core_grid = input_tensor.device()->compute_with_storage_grid_size();

    // output
    TT_FATAL(this->output_mem_config.is_sharded() && this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
    uint32_t num_cores = core_grid.x * core_grid.y;
    // Support maximum 32 heads for now
    TT_FATAL(this->num_q_heads <= 32);
    // 1 User Per Core Max and 32 users for now
    TT_FATAL(num_cores >= 32, "Need at least 32 cores for decode");
    TT_FATAL(this->num_q_heads >= this->num_kv_heads);
}

std::vector<Shape> NlpCreateHeadsDecode::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    std::vector<Shape> output_shape_vec;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    auto batch = input_tensor.get_shape()[2];
    auto head_dim = this->head_dim;

    // pad up to nearest multiple of TILE_HEIGHT for num_q_heads and num_kv_heads
    auto num_q_heads_padded = (this->num_q_heads / TILE_HEIGHT + 1) * TILE_HEIGHT;
    auto num_kv_heads_padded = (this->num_kv_heads / TILE_HEIGHT + 1) * TILE_HEIGHT;

    const Shape q_output_shape = {input_shape[0], batch, num_q_heads_padded, head_dim};
    const Shape v_output_shape = {input_shape[0], batch, num_kv_heads_padded, head_dim};
    const Shape k_output_shape = v_output_shape;
    output_shape_vec = {q_output_shape, k_output_shape, v_output_shape};

    return output_shape_vec;
}

std::vector<Tensor> NlpCreateHeadsDecode::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    auto output_shapes = this->compute_output_shapes(input_tensors);
    const auto& q_output_shape = output_shapes[0];

    auto batch = q_output_shape[1];
    auto num_q_heads_padded = (this->num_q_heads / TILE_HEIGHT + 1) * TILE_HEIGHT;
    auto num_kv_heads_padded = (this->num_kv_heads / TILE_HEIGHT + 1) * TILE_HEIGHT;
    auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
    auto q_shard_grid = num_cores_to_corerange_set(batch, core_grid, true);
    ShardSpec q_shard_spec{q_shard_grid, {num_q_heads_padded, this->head_dim}};
    auto q_mem_config = this->output_mem_config;
    q_mem_config.shard_spec = q_shard_spec;
    auto kv_shard_grid = num_cores_to_corerange_set(batch, core_grid, true);
    ShardSpec kv_shard_spec{kv_shard_grid, {num_kv_heads_padded, this->head_dim}};
    auto kv_mem_config = this->output_mem_config;
    kv_mem_config.shard_spec = kv_shard_spec;
    return {
        create_device_tensor(output_shapes[0], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), q_mem_config),
        create_device_tensor(output_shapes[1], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), kv_mem_config),
        create_device_tensor(output_shapes[2], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), kv_mem_config)
    };
}

operation::ProgramWithCallbacks NlpCreateHeadsDecode::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    return  multi_core_nlp_create_qkv_heads_decode(input_tensor, this->num_q_heads, this->num_kv_heads, this->head_dim, output_tensors, compute_with_storage_grid_size);
}

// NLP ConcatHeads op for decode
void NlpConcatHeadsDecode::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    // input tensor and shape
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16, "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE);
    TT_FATAL(input_shape[0] == 1, "seqlen=1 for decode");
    TT_FATAL(input_shape[1] <= 32, "currently only support less than 32 users");
    TT_FATAL(input_shape[2] == 32, "currently only support 32 padded heads");
    TT_FATAL(input_shape[2] >= this->num_heads, "head_dim must be multiple of TILE_WIDTH");

    // input tensor shard spec
    TT_FATAL(input_tensor.is_sharded());
    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
    auto shard_spec = input_tensor.shard_spec().value();
    TT_FATAL(shard_spec.shape[1] == input_tensor.get_legacy_shape()[-1]);
    TT_FATAL(shard_spec.shape[0] == input_tensor.get_legacy_shape()[-2]);
    auto shard_grid = shard_spec.grid.bounding_box().grid_size();
    auto num_cores = shard_grid.x * shard_grid.y;
    TT_FATAL(num_cores == input_shape[1], "num_cores must be equal to num users");
}

std::vector<Shape> NlpConcatHeadsDecode::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    auto num_heads = this->num_heads;
    auto sequence_length = input_shape[0];
    auto batch = input_shape[1];
    auto head_dim = input_shape[3];
    // pad batch to 32 if necessary
    if (batch < 32) {
        batch = 32;
    }

    auto hidden_dim = num_heads * head_dim;

    const Shape output_shape = {sequence_length, 1, batch, hidden_dim};
    return {output_shape};
}

std::vector<Tensor> NlpConcatHeadsDecode::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto num_heads = this->num_heads;
    const auto input_shape = input_tensor.get_legacy_shape();
    auto sequence_length = input_shape[0];
    auto head_dim = input_shape[3];
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    auto batch = output_shape[2];

    auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
    auto shard_grid = num_cores_to_corerange_set(num_heads, core_grid, true);
    ShardSpec shard_spec{shard_grid, {batch, head_dim}};
    auto mem_config = tt::tt_metal::MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1};
    mem_config.shard_spec = shard_spec;

    return {create_device_tensor(output_shape, input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), mem_config)};
}

operation::ProgramWithCallbacks NlpConcatHeadsDecode::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    return  multi_core_nlp_concat_heads_decode(input_tensor, output_tensor, compute_with_storage_grid_size);
}

// NLP KV Cache Unpad To Sharded op
void NlpKVCacheLoadSlice::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE);

    for (uint32_t i = 0; i < input_tensor_a.get_legacy_shape().rank(); i++) {
        TT_FATAL(this->output_tensor_start[i] < input_tensor_a.get_legacy_shape()[i]);
        TT_FATAL(this->output_tensor_end[i] < input_tensor_a.get_legacy_shape()[i]);

        // Check if start shape is <= end shape
        TT_FATAL(this->output_tensor_start[i] <= this->output_tensor_end[i]);
    }

    Shape output_tensor_shape = this->compute_output_shapes(input_tensors)[0];
    auto num_dims = input_tensor_a.get_legacy_shape().rank();
    TT_FATAL(num_dims == 4, "Input tensor must be 4D");
    const auto input_shape = input_tensor_a.get_legacy_shape();
    auto dim0 = input_shape[0];
    auto dim1 = input_shape[1];
    auto fused_batch_heads = dim0 * dim1;
    auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
    // Need at least fused_batch_heads cores to unpad into sharded tensor
    TT_FATAL(fused_batch_heads <= core_grid.x * core_grid.y);
    TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);
    TT_FATAL((output_tensor_shape[-2] % TILE_HEIGHT == 0) &&
                (this->output_tensor_start[-2] % TILE_HEIGHT == 0),
            "Can only unpad tilized tensor with full tiles");
    TT_FATAL((output_tensor_shape[-1] % TILE_WIDTH == 0) &&
                (this->output_tensor_start[-1] % TILE_WIDTH == 0),
            "Can only unpad tilized tensor with full tiles");
}
std::vector<Shape> NlpKVCacheLoadSlice::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    std::vector<uint32_t> out_shape;
    auto rank = input_tensors[0].get_legacy_shape().rank();
    out_shape.reserve(rank);
    for (uint32_t i = 0; i < rank; i++) {
        out_shape.push_back(this->output_tensor_end[i] - this->output_tensor_start[i] + 1);
    }
    Shape output_tensor_shape(out_shape);
    return {output_tensor_shape};
}
std::vector<Tensor> NlpKVCacheLoadSlice::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto input_shape = input_tensor_a.get_legacy_shape();
    auto dim0 = input_shape[0];
    auto dim1 = input_shape[1];
    auto unpad_length = this->output_tensor_end[2] - this->output_tensor_start[2] + 1;
    auto head_dim = input_shape[3];
    auto fused_batch_heads = dim0 * dim1;

    auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
    auto shard_grid = num_cores_to_corerange_set(fused_batch_heads, core_grid, true);
    ShardSpec shard_spec{shard_grid, {unpad_length, head_dim}};
    auto mem_config = tt::tt_metal::MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1};
    mem_config.shard_spec = shard_spec;

    return {create_device_tensor(
        this->compute_output_shapes(input_tensors).at(0),
        input_tensor_a.get_dtype(),
        input_tensor_a.get_layout(),
        input_tensor_a.device(),
        mem_config
        )};
}
operation::ProgramWithCallbacks NlpKVCacheLoadSlice::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return multi_core_nlp_kv_cache_load_slice(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end);
}

} // namespace tt_metal

} // namespace tt
