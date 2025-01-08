// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/untilize_with_halo_v2/device/untilize_with_halo_v2_program_factory.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/sliding_window/halo/device/halo_device_operation.hpp"

namespace ttnn::operations::sliding_window::halo {

using namespace tt::tt_metal;

thread_local std::unordered_map<std::size_t, std::uint32_t>
    HaloDeviceOperation::sliding_window_max_out_nsticks_per_core = {};

void HaloDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    // validate input data tensor
    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        // skip the untilize, only do halo
        log_debug(tt::LogOp, "Input is ROW_MAJOR, no need to untilize.");
    } else {
        TT_FATAL(input_tensor.volume() % tt::constants::TILE_HW == 0, "Error");
    }
    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
        "Only height, width or block sharded tensors are supported.");
    TT_FATAL(input_tensor.shard_spec().has_value(), "Shard spec should not be empty");
}

std::vector<TensorSpec> HaloDeviceOperation::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.get_legacy_shape();
    ttnn::SimpleShape output_shape = ttnn::SimpleShape(input_shape.to_array_4D());

    uint32_t nbatch = input_shape[0];
    uint32_t total_nsticks = config_.num_cores_nhw * max_out_nsticks_per_core_;

    // output_shape[0] remains same
    // output_shape[1] remains same
    // output_shape[2] changes
    // output_shape[3] remains same
    output_shape[2] = (uint32_t)std::ceil((float)total_nsticks / nbatch);

    log_debug(
        tt::LogOp, "output_shape: [{} {} {} {}]", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    log_debug(tt::LogOp, "max_out_nsticks_per_core: {}", max_out_nsticks_per_core_);
    log_debug(tt::LogOp, "num_cores_nhw: {}", config_.num_cores_nhw);

    const auto& input_tensor = input_tensors.at(0);
    DataType output_dtype =
        input_tensor.get_dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.get_dtype();

    TT_FATAL(
        input_tensor.memory_config().memory_layout == output_memory_config_.memory_layout,
        "{} {}",
        input_tensor.memory_config(),
        output_memory_config_);

    if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto input_core_range = *(input_tensor.memory_config().shard_spec->grid.ranges().begin());
        auto output_core_range = *(output_memory_config_.shard_spec->grid.ranges().begin());
        auto input_core_w = input_core_range.end_coord.y - input_core_range.start_coord.y + 1;
        auto output_core_w = output_core_range.end_coord.y - output_core_range.start_coord.y + 1;
        TT_FATAL(input_core_w == output_core_w, "Error");
    }

    auto out_mem_config = output_memory_config_;
    out_mem_config.shard_spec->shape[0] = tt::div_up(output_shape[0] * output_shape[2], config_.num_cores_nhw);
    out_mem_config.shard_spec->shape[1] = input_tensor.memory_config().shard_spec->shape[1];
    out_mem_config.shard_spec->halo = true;
    return {TensorSpec(
        output_shape,
        TensorLayout::fromLegacyPaddedShape(
            output_dtype, PageConfig(Layout::ROW_MAJOR), out_mem_config, ttnn::Shape(output_shape)))};
}

operation::ProgramWithCallbacks HaloDeviceOperation::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    const auto& input_tensor = inputs.at(0);
    auto& output_tensor = outputs.at(0);
    auto device = input_tensor.device();

    bool is_in_tiled = input_tensor.get_layout() == Layout::TILE;
    bool is_block_sharded = input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED;

    auto pad_metadata = sliding_window::generate_pad_metadata(config_);
    auto op_trace_metadata = sliding_window::generate_op_trace_metadata(config_);
    auto shard_boundaries = sliding_window::generate_shard_boundaries(config_, op_trace_metadata);
    auto tensor_metadata = sliding_window::generate_tensor_metadata(
        pad_metadata, config_, reshard_num_cores_nhw_, is_in_tiled || is_out_tiled_);
    auto kernel_config = sliding_window::generate_halo_kernel_config_tensors(
        tensor_metadata, shard_boundaries, is_block_sharded, transpose_mcast_, remote_read_, device);

    const auto& pad_config = std::get<0>(kernel_config);
    const auto& local_config = std::get<1>(kernel_config);
    const auto& remote_config = std::get<2>(kernel_config);

    auto pad_config_tensor =
        sliding_window::construct_on_host_config_tensor(pad_config, this->config_, this->parallel_config_);
    auto local_config_tensor =
        sliding_window::construct_on_host_config_tensor(local_config, this->config_, this->parallel_config_);
    auto remote_config_tensor =
        sliding_window::construct_on_host_config_tensor(remote_config, this->config_, this->parallel_config_);

    auto pad_config_device_tensor =
        sliding_window::move_config_tensor_to_device(pad_config_tensor, parallel_config_, is_block_sharded, device);
    auto local_config_device_tensor =
        sliding_window::move_config_tensor_to_device(local_config_tensor, parallel_config_, is_block_sharded, device);
    auto remote_config_device_tensor =
        sliding_window::move_config_tensor_to_device(remote_config_tensor, parallel_config_, is_block_sharded, device);

    Program program = CreateProgram();

    tt::tt_metal::detail::AddConfigBuffer(program, pad_config_device_tensor.device_buffer());
    tt::tt_metal::detail::AddConfigBuffer(program, local_config_device_tensor.device_buffer());
    tt::tt_metal::detail::AddConfigBuffer(program, remote_config_device_tensor.device_buffer());

    return {data_movement::detail::untilize_with_halo_multi_core_v2(
        program,
        input_tensor,
        pad_val_,
        config_.num_cores_nhw,
        max_out_nsticks_per_core_,
        pad_config_device_tensor,
        local_config_device_tensor,
        remote_config_device_tensor,
        remote_read_,
        transpose_mcast_,
        output_tensor)};
}

Tensor halo_op(
    const Tensor& input_tensor,
    const SlidingWindowConfig& config,
    uint32_t pad_val,
    bool remote_read,
    bool transpose_mcast,
    uint32_t reshard_num_cores_nhw,
    const MemoryConfig& output_memory_config,
    bool is_out_tiled) {
    TT_FATAL(input_tensor.memory_config().is_sharded(), "Halo expects sharded input tensor");
    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
        "Only height, width or block sharded tensors are supported.");
    // NOTE: for HEIGHT_SHARDED, ncores_nhw == ncores
    //       for BLOCK_SHARDED, ncores_nhw is just the ncores along height dim (last tensor dim is split along width)
    bool is_block_sharded = input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED;
    auto halo_func =
        [config,
         pad_val,
         remote_read,
         is_block_sharded,
         transpose_mcast,
         reshard_num_cores_nhw,
         output_memory_config,
         is_out_tiled](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
        auto input_tensor = input_tensors.at(0);

        auto device = input_tensor.device();

        auto sliding_window_hash = config.get_hash();
        if (!HaloDeviceOperation::sliding_window_max_out_nsticks_per_core.contains(sliding_window_hash)) {
            auto op_trace_metadata = sliding_window::generate_op_trace_metadata(config);
            auto shard_boundaries = sliding_window::generate_shard_boundaries(config, op_trace_metadata);
            HaloDeviceOperation::sliding_window_max_out_nsticks_per_core.emplace(
                sliding_window_hash, sliding_window::generate_max_out_nsticks_per_core(shard_boundaries));
        }

        uint32_t max_out_nsticks_per_core =
            HaloDeviceOperation::sliding_window_max_out_nsticks_per_core.at(sliding_window_hash);
        ParallelConfig p_config;
        p_config.grid = input_tensor.shard_spec().value().grid;
        p_config.shard_scheme = input_tensor.memory_config().memory_layout;
        p_config.shard_orientation = input_tensor.shard_spec().value().orientation;

        return operation::run(
            HaloDeviceOperation{
                .config_ = config,
                .parallel_config_ = p_config,
                .pad_val_ = pad_val,
                .remote_read_ = remote_read,
                .transpose_mcast_ = transpose_mcast,
                .reshard_num_cores_nhw_ = reshard_num_cores_nhw,
                .max_out_nsticks_per_core_ = max_out_nsticks_per_core,
                .output_memory_config_ = output_memory_config,
                .is_out_tiled_ = is_out_tiled},
            {input_tensor});
    };

    std::vector<Tensor> output_tensors = {
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}, {}))};
    operation::launch_op(halo_func, {input_tensor}, output_tensors);

    return output_tensors.at(0);
}

}  // namespace ttnn::operations::sliding_window::halo
