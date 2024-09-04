// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_op.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/run_operation.hpp"
#include "untilize_with_unpadding_program_factory.hpp"

namespace ttnn::operations::data_movement {

void UntilizeWithUnpadding::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "Can only untilize tile major data");

    TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);
    for (uint32_t i = 0; i < input_tensor_a.get_legacy_shape().rank(); i++) {
        TT_FATAL(input_tensor_a.get_legacy_shape()[i] > 0);
        TT_FATAL(this->output_tensor_end[i] < input_tensor_a.get_legacy_shape()[i]);
    }

    TT_FATAL(((this->output_tensor_end[-1] + 1) % 2 == 0), "Can only unpad to row major tensor of even width");

    if (input_tensor_a.memory_config().is_sharded()) {
        if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1);
            TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
            TT_FATAL(
                input_tensor_a.volume() /
                        (input_tensor_a.get_legacy_shape()[-2] * input_tensor_a.get_legacy_shape()[-1]) ==
                    1,
                "Can only write unbatched output interleaved");
        } else if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            if (output_mem_config.is_sharded()) {
                TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
            }
            // What else?
        } else if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            auto output_shape = this->compute_output_shapes(input_tensors).at(0);
            // Minor host code changes required to remove this restriction
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1);
            for (uint32_t i = 0; i < output_shape.rank() - 2; i++) {
                TT_FATAL(input_tensor_a.get_legacy_shape()[i] == output_shape[i]);
            }
            if (output_mem_config.is_sharded()) {
                TT_FATAL(this->output_mem_config.memory_layout == input_tensor_a.memory_config().memory_layout);
                TT_FATAL(
                    input_tensor_a.get_legacy_shape()[-1] == output_shape[-1] ||
                    (tt::div_up(output_shape[-1], input_tensor_a.shard_spec().value().shape[1]) ==
                     input_tensor_a.shard_spec().value().grid.num_cores()));
            } else {
                TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
                TT_FATAL(
                    input_tensor_a.volume() /
                            (input_tensor_a.get_legacy_shape()[-2] * input_tensor_a.get_legacy_shape()[-1]) ==
                        1,
                    "Can only write unbatched output interleaved");
                TT_FATAL(
                    input_tensor_a.get_legacy_shape()[-1] - output_shape[-1] <
                    input_tensor_a.shard_spec().value().shape[1]);
            }
        } else {
            TT_FATAL(false, "Unsupported sharding scheme");
        }
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<tt::tt_metal::Shape> UntilizeWithUnpadding::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    std::vector<uint32_t> out_shape;
    auto rank = input_tensors[0].get_legacy_shape().rank();
    out_shape.reserve(rank);
    for (uint32_t i = 0; i < rank; i++) {
        out_shape.push_back(this->output_tensor_end[i] + 1);
    }
    tt::tt_metal::Shape output_tensor_shape(out_shape);
    return {output_tensor_shape};
}

std::vector<Tensor> UntilizeWithUnpadding::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    DataType output_dtype =
        input_tensor_a.get_dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor_a.get_dtype();
    if (input_tensor_a.memory_config().is_sharded() && this->output_mem_config.is_sharded()) {
        auto output_shape = this->compute_output_shapes(input_tensors).at(0);
        uint32_t fused_height = tt::tt_metal::compute_volume(output_shape) / output_shape[-1];
        uint32_t num_cores = input_tensor_a.shard_spec().value().num_cores();
        std::array<uint32_t, 2> shard_shape;
        ShardSpec shard_spec = input_tensor_a.shard_spec().value();
        if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            shard_shape = {tt::div_up(fused_height, num_cores), output_shape[-1]};
        } else {
            shard_shape = {fused_height, shard_spec.shape[1]};
        }
        shard_spec.shape = shard_shape;
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = shard_spec;
        return {create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            output_dtype,
            Layout::ROW_MAJOR,
            input_tensor_a.device(),
            mem_config)};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, output_dtype, Layout::ROW_MAJOR, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks UntilizeWithUnpadding::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (input_tensors.at(0).memory_config().is_sharded() || this->use_multicore) {
        return detail::untilize_with_unpadding_multi_core(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    } else {
        return detail::untilize_with_unpadding_single_core(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }
}

}  // namespace ttnn::operations::data_movement
