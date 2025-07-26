// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_op.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/run_operation.hpp"
#include "untilize_with_unpadding_program_factory.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void UntilizeWithUnpadding::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    TT_FATAL(input_tensor_a.physical_volume() % tt::constants::TILE_HW == 0, "Error");

    if (input_tensor_a.memory_config().is_sharded()) {
        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1, "Error");
            TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
            TT_FATAL(
                input_tensor_a.physical_volume() /
                        (input_tensor_a.padded_shape()[-2] * input_tensor_a.padded_shape()[-1]) ==
                    1,
                "Can only write unbatched output interleaved");
        } else if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            if (output_mem_config.is_sharded()) {
                TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
            }
            // What else?
        } else if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            auto output_shape = this->compute_output_specs(input_tensors).at(0).padded_shape();
            for (uint32_t i = 0; i < output_shape.rank() - 2; i++) {
                TT_FATAL(input_tensor_a.padded_shape()[i] == output_shape[i], "Error");
            }
            if (output_mem_config.is_sharded()) {
                TT_FATAL(
                    this->output_mem_config.memory_layout() == input_tensor_a.memory_config().memory_layout(), "Error");
                TT_FATAL(
                    input_tensor_a.padded_shape()[-1] == output_shape[-1] ||
                        (tt::div_up(output_shape[-1], input_tensor_a.shard_spec().value().shape[1]) ==
                         input_tensor_a.shard_spec().value().grid.num_cores()),
                    "Error");
            } else {
                TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
                TT_FATAL(
                    input_tensor_a.physical_volume() /
                            (input_tensor_a.padded_shape()[-2] * input_tensor_a.padded_shape()[-1]) ==
                        1,
                    "Can only write unbatched output interleaved");
                TT_FATAL(
                    input_tensor_a.padded_shape()[-1] - output_shape[-1] < input_tensor_a.shard_spec().value().shape[1],
                    "Error");
            }
        } else {
            TT_THROW("Unsupported sharding scheme");
        }
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
        TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
    }
}

std::vector<ttnn::TensorSpec> UntilizeWithUnpadding::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    SmallVector<uint32_t> out_shape;
    const auto& input_tensor_a = input_tensors.at(0);
    size_t rank = input_tensor_a.logical_shape().rank();
    out_shape.reserve(rank);
    for (uint32_t i = 0; i < rank; i++) {
        out_shape.push_back(this->output_tensor_end[i] + 1);
    }
    Shape output_shape(std::move(out_shape));

    DataType output_dtype = input_tensor_a.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor_a.dtype();
    if (input_tensor_a.memory_config().is_sharded() && this->output_mem_config.is_sharded()) {
        uint32_t fused_height = output_shape.volume() / output_shape[-1];
        uint32_t num_cores = input_tensor_a.shard_spec().value().num_cores();
        std::array<uint32_t, 2> shard_shape;
        ShardSpec shard_spec = input_tensor_a.shard_spec().value();
        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            const auto tile = input_tensor_a.tensor_spec().tile();
            uint32_t tile_height = tile.get_height();
            uint32_t shard_idx0 = tt::round_up(tt::div_up(fused_height, num_cores), tile_height);
            shard_shape = {shard_idx0, output_shape[-1]};
        } else {
            shard_shape = {fused_height, shard_spec.shape[1]};
        }
        shard_spec.shape = shard_shape;
        auto mem_config = this->output_mem_config.with_shard_spec(shard_spec);
        return {TensorSpec(output_shape, TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), mem_config))};
    }

    return {TensorSpec(output_shape, TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), output_mem_config))};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>>
UntilizeWithUnpadding::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    uint32_t single_tile_size = tile_width * tile_height * input_tensor.element_size();
    uint32_t num_tiles = std::ceil((float)input_tensor.physical_volume() / (float)single_tile_size);
    int compute_cycles = 0;
    const int max_tiles_per_row = 8;
    const int latency_untilize = 390;      // measured latency for untilize_block
    const int latency_pack_untilize = 80;  // measured latency for pack_untilize_block
    if (std::ceil((float)input_tensor.padded_shape()[-1] / (float)tile_width) <= max_tiles_per_row) {
        compute_cycles = num_tiles * latency_pack_untilize;
    } else {
        compute_cycles = num_tiles * latency_untilize;
    }
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor, false, compute_cycles);
    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> result(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}

operation::ProgramWithCallbacks UntilizeWithUnpadding::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (input_tensors.at(0).memory_config().is_sharded()) {
        return detail::untilize_with_unpadding_multi_core_sharded(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }
    if (!this->use_multicore) {
        return detail::untilize_with_unpadding_single_core(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }
    if (!this->enough_space_height) {
        return detail::untilize_with_unpadding_multi_core_block_interleaved(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }
    return detail::untilize_with_unpadding_multi_core_interleaved(
        input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
}

}  // namespace ttnn::operations::data_movement
