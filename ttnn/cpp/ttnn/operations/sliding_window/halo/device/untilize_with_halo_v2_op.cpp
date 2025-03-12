// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_halo_v2_op.hpp"

#include "ttnn/run_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "untilize_with_halo_v2_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void UntilizeWithHaloV2::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);

    // validate input data tensor
    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        // skip the untilize, only do halo
        log_debug(tt::LogOp, "Input is ROW_MAJOR, no need to untilize.");
    } else {
        TT_FATAL(input_tensor.volume() % TILE_HW == 0, "Error");
    }
    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout);
    TT_FATAL(input_tensor.shard_spec().has_value(), "Error");
}

std::vector<ttnn::TensorSpec> UntilizeWithHaloV2::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input_shape = input_tensor.get_padded_shape();
    auto output_shape = input_shape;

    uint32_t nbatch = input_shape[0];
    uint32_t total_nsticks = ncores_nhw_ * max_out_nsticks_per_core_;

    // output_shape[0] remains same
    // output_shape[1] remains same
    // output_shape[2] changes
    // output_shape[3] remains same
    output_shape[2] = (uint32_t)std::ceil((float)total_nsticks / nbatch);

    log_debug(
        tt::LogOp, "output_shape: [{} {} {} {}]", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    log_debug(tt::LogOp, "max_out_nsticks_per_core: {}", max_out_nsticks_per_core_);
    log_debug(tt::LogOp, "ncores_nhw: {}", ncores_nhw_);

    DataType output_dtype =
        input_tensor.get_dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.get_dtype();
    TT_FATAL(
        input_tensor.memory_config().memory_layout == out_mem_config_.memory_layout,
        "{} {}",
        input_tensor.memory_config(),
        out_mem_config_);
    if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto input_core_range = *(input_tensor.memory_config().shard_spec->grid.ranges().begin());
        auto output_core_range = *(out_mem_config_.shard_spec->grid.ranges().begin());
        auto input_core_w = input_core_range.end_coord.y - input_core_range.start_coord.y + 1;
        auto output_core_w = output_core_range.end_coord.y - output_core_range.start_coord.y + 1;
        TT_FATAL(input_core_w == output_core_w, "Error");
    }

    auto out_mem_config = out_mem_config_;
    out_mem_config.shard_spec->shape[0] = tt::div_up(output_shape[0] * output_shape[2], ncores_nhw_);
    out_mem_config.shard_spec->shape[1] = input_tensor.memory_config().shard_spec->shape[1];
    return {TensorSpec(output_shape, TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), out_mem_config))};
}

operation::ProgramWithCallbacks UntilizeWithHaloV2::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& padding_config1 = input_tensors.at(1);
    const auto& padding_config2 = input_tensors.at(2);
    const auto& local_config1 = input_tensors.at(3);
    const auto& local_config2 = input_tensors.at(4);
    const auto& remote_config1 = input_tensors.at(5);
    const auto& remote_config2 = input_tensors.at(6);
    auto& output_tensor = output_tensors.at(0);

    Program program = CreateProgram();

    return {detail::untilize_with_halo_multi_core_v2(
        program,
        input_tensor,
        pad_val_,
        ncores_nhw_,
        max_out_nsticks_per_core_,
        padding_config1,
        padding_config2,
        local_config1,
        local_config2,
        remote_config1,
        remote_config2,
        remote_read_,
        transpose_mcast_,
        output_tensor,
        /*capture_buffers=*/false)};
}

}  // namespace ttnn::operations::data_movement
