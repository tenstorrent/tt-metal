// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads_device_operation.hpp"

#include "split_query_key_value_and_split_heads_program_factory.hpp"

namespace ttnn::operations::experimental::transformer {

void SplitFusedQKVAndSplitHeadsDeviceOperation::validate_with_output_tensors(const std::vector<Tensor>& input_tensors,
                                                                            const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.get_legacy_shape()[0];
    // TODO: See issue #1744
    TT_FATAL((input_tensor.get_legacy_shape() == tt::tt_metal::Shape({batch_size, 1, 384, 3072})), "Unsupported input shape");
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
             bbox.end_coord.y < this->compute_with_storage_grid_size.y), "Error");
        TT_FATAL(input_tensor.shard_spec().value().grid.ranges().size() == 1, "Error");
        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED, "Error");
    }

    if (!output_tensors.empty()) {
        TT_FATAL(
            output_tensors.size() == 3, "Must have 3 output tensors");
    }
}

std::vector<tt::tt_metal::Shape> SplitFusedQKVAndSplitHeadsDeviceOperation::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.get_legacy_shape()[0];
    uint32_t num_heads = this->num_heads;
    uint32_t num_output_tensors = 3;
    uint32_t M = input_tensor.get_legacy_shape()[2];                                    // 384
    uint32_t K = input_tensor.get_legacy_shape()[-1] / num_output_tensors / num_heads;  // 64
    return {
        tt::tt_metal::Shape{batch_size, this->num_heads, M, K},
        tt::tt_metal::Shape{batch_size, this->num_heads, K, M},
        tt::tt_metal::Shape{batch_size, this->num_heads, M, K}};
}

std::vector<Tensor> SplitFusedQKVAndSplitHeadsDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (output_tensors.size() == 3 && output_tensors[0].has_value() && output_tensors[1].has_value() && output_tensors[2].has_value()) {
        return {output_tensors.at(0).value(), output_tensors.at(1).value(), output_tensors.at(2).value()};
    }
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
            tt::tt_metal::Shape{batch, num_heads, M, K},
            input_tensor.get_dtype(),
            Layout::TILE,
            input_tensor.device(),
            mem_config_qv);
        auto out_tensor_k = create_device_tensor(
            tt::tt_metal::Shape{batch, num_heads, K, M}, input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), mem_config_k);
        auto out_tensor_v = create_device_tensor(
            tt::tt_metal::Shape{batch, num_heads, M, K},
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

operation::ProgramWithCallbacks SplitFusedQKVAndSplitHeadsDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto device_compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    TT_ASSERT(
        (this->compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x &&
         this->compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y),
        "Unsupported grid shape");

    if (input_tensor.is_sharded()) {
        return detail::multi_core_split_query_key_value_and_split_heads_sharded(
            input_tensor, output_tensors, this->compute_with_storage_grid_size);
    } else {
        return detail::multi_core_split_query_key_value_and_split_heads(
            input_tensor, output_tensors, this->compute_with_storage_grid_size);
    }
}

}  // namespace ttnn::operations::experimental::transformer
