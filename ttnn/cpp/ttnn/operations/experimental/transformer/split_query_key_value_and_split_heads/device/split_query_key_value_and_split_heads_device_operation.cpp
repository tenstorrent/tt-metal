// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads_device_operation.hpp"

#include "split_query_key_value_and_split_heads_program_factory.hpp"

namespace ttnn::operations::experimental::transformer {

void SplitFusedQKVAndSplitHeadsDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.padded_shape()[0];
    // TODO: See issue #1744
    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");

    if (!input_tensor.is_sharded()) {
        TT_FATAL(batch_size >= 7 && batch_size <= 9, "Input batch size must be between 2 to 9 for bert large TM ops!");
    } else {
        auto bbox = input_tensor.shard_spec().value().grid.bounding_box();
        TT_FATAL(
            (bbox.end_coord.x < this->compute_with_storage_grid_size.x &&
             bbox.end_coord.y < this->compute_with_storage_grid_size.y),
            "Error");
        TT_FATAL(input_tensor.shard_spec().value().grid.ranges().size() == 1, "Error");
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED, "Error");
    }

    if (!output_tensors.empty()) {
        TT_FATAL(output_tensors.size() == 3, "Must have 3 output tensors");
    }
}

std::vector<ttnn::TensorSpec> SplitFusedQKVAndSplitHeadsDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    using tt::tt_metal::Layout;
    using tt::tt_metal::PageConfig;
    using tt::tt_metal::TensorLayout;

    if (output_tensors.size() == 3 && output_tensors[0].has_value() && output_tensors[1].has_value() &&
        output_tensors[2].has_value()) {
        return {
            output_tensors.at(0)->tensor_spec(),
            output_tensors.at(1)->tensor_spec(),
            output_tensors.at(2)->tensor_spec()};
    }

    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.padded_shape()[0];
    uint32_t num_heads = this->num_heads;
    uint32_t num_output_tensors = 3;
    uint32_t M = input_tensor.padded_shape()[2];                                    // 384
    uint32_t K = input_tensor.padded_shape()[-1] / num_output_tensors / num_heads;  // 64

    if (input_tensor.is_sharded()) {
        // core range
        CoreRangeSet all_cores = input_tensor.shard_spec().value().grid;
        tt::tt_metal::ShardOrientation shard_orientation = input_tensor.shard_spec().value().orientation;
        auto bbox = all_cores.bounding_box();
        uint32_t num_M_cores = shard_orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR ? bbox.end_coord.x + 1
                                                                                              : bbox.end_coord.y + 1;
        // shard spec
        uint32_t per_core_M_qv = (num_heads / num_M_cores) * M;  // 768
        uint32_t per_core_N_qv = K;                              // 64
        auto shard_spec_qv = tt::tt_metal::ShardSpec{all_cores, {per_core_M_qv, per_core_N_qv}, shard_orientation};
        uint32_t per_core_M_k = (num_heads / num_M_cores) * K;  // 128
        uint32_t per_core_N_k = M;                              // 384
        auto shard_spec_k = tt::tt_metal::ShardSpec{all_cores, {per_core_M_k, per_core_N_k}, shard_orientation};
        // create sharded tensors
        auto mem_config_qv = this->output_mem_config.with_shard_spec(shard_spec_qv);
        auto mem_config_k = this->output_mem_config.with_shard_spec(shard_spec_k);
        auto out_tensor_q = TensorSpec(
            Shape({batch_size, num_heads, M, K}),
            TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), mem_config_qv));
        auto out_tensor_k = TensorSpec(
            Shape({batch_size, num_heads, K, M}),
            TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), mem_config_k));
        auto out_tensor_v = TensorSpec(
            Shape({batch_size, num_heads, M, K}),
            TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), mem_config_qv));
        return {out_tensor_q, out_tensor_k, out_tensor_v};
    }

    TensorLayout layout(input_tensor.dtype(), PageConfig(Layout::TILE), output_mem_config);
    return {
        TensorSpec(Shape({batch_size, this->num_heads, M, K}), layout),
        TensorSpec(Shape({batch_size, this->num_heads, K, M}), layout),
        TensorSpec(Shape({batch_size, this->num_heads, M, K}), layout),
    };
}

std::vector<Tensor> SplitFusedQKVAndSplitHeadsDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    auto specs = compute_output_specs(input_tensors, output_tensors);
    return {
        create_device_tensor(specs[0], input_tensors.at(0).device()),
        create_device_tensor(specs[1], input_tensors.at(0).device()),
        create_device_tensor(specs[2], input_tensors.at(0).device()),
    };
}

tt::tt_metal::operation::ProgramWithCallbacks SplitFusedQKVAndSplitHeadsDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

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
