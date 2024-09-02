// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_op.hpp"

#include "ttnn/run_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "untilize_program_factory.hpp"

namespace ttnn::operations::data_movement {

namespace untilize_helpers {
uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks) {
    int32_t ncores_x = grid_size.x;
    int32_t ncores_y = grid_size.y;
    int32_t ncores = ncores_x * ncores_y;
    if (nblocks <= ncores) {
        ncores = nblocks;
    } else {
        uint32_t nblocks_per_core = ceil((float)nblocks / ncores);
        ncores = ceil((float)nblocks / nblocks_per_core);
    }
    return ncores;
}
}  // namespace untilize_helpers

void Untilize::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to untilize need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to untilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "Can only untilize tile major data");

    TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);

    if (input_tensor_a.memory_config().is_sharded()) {
        if (this->output_mem_config.is_sharded()) {
            TT_FATAL(this->output_mem_config.memory_layout == input_tensor_a.memory_config().memory_layout);
        }
        if (input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1);
        }
        TT_FATAL(this->use_multicore == true);
    } else if (this->output_mem_config.is_sharded()) {
        TT_FATAL(this->use_multicore == true);
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        uint32_t ntiles = input_tensor_a.volume() / TILE_HW;
        uint32_t ntiles_per_block = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
        uint32_t nblocks = ceil((float)ntiles / ntiles_per_block);
        auto num_cores =
            untilize_helpers::get_num_cores(input_tensor_a.device()->compute_with_storage_grid_size(), nblocks);
        uint32_t fused_height = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1];
        TT_FATAL(fused_height % num_cores == 0);
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<tt::tt_metal::Shape> Untilize::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return {input_tensor_a.get_legacy_shape()};
}

std::vector<Tensor> Untilize::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);
    DataType output_dtype =
        input_tensor.get_dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.get_dtype();
    if (output_mem_config.is_sharded()) {
        if (input_tensor.memory_config().is_sharded()) {
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = input_tensor.memory_config().shard_spec;
            return {create_device_tensor(
                this->compute_output_shapes(input_tensors).at(0),
                output_dtype,
                Layout::ROW_MAJOR,
                input_tensor.device(),
                mem_config)};
        } else {
            uint32_t ntiles = input_tensor.volume() / TILE_HW;
            uint32_t ntiles_per_block = input_tensor.get_legacy_shape()[-1] / TILE_WIDTH;
            uint32_t nblocks = ceil((float)ntiles / ntiles_per_block);
            auto num_cores =
                untilize_helpers::get_num_cores(input_tensor.device()->compute_with_storage_grid_size(), nblocks);
            auto shard_grid =
                tt::tt_metal::num_cores_to_corerange_set(num_cores, input_tensor.device()->compute_with_storage_grid_size(), true);
            uint32_t fused_height = input_tensor.volume() / input_tensor.get_legacy_shape()[-1];
            std::array<uint32_t, 2> shard_shape = {fused_height / num_cores, input_tensor.get_legacy_shape()[-1]};
            ShardSpec shard_spec{shard_grid, shard_shape, ShardOrientation::ROW_MAJOR};
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = shard_spec;
            return {create_device_tensor(
                this->compute_output_shapes(input_tensors).at(0),
                output_dtype,
                Layout::ROW_MAJOR,
                input_tensor.device(),
                mem_config)};
        }
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, output_dtype, Layout::ROW_MAJOR, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Untilize::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (this->use_multicore) {
        return detail::untilize_multi_core(input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    } else {
        return detail::untilize_single_core(input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }
}

}  // namespace ttnn::operations::data_movement
