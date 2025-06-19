// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/sliding_window/halo/device/halo_device_operation.hpp"
#include <array>

namespace ttnn::operations::sliding_window::halo {

using namespace tt::tt_metal;

thread_local std::unordered_map<std::size_t, std::uint32_t>
    HaloDeviceOperation::sliding_window_max_out_nsticks_per_core = {};

// TODO: Look into increasing this to tradeoff some L1 for performance (#19980)
constexpr int UNTILIZE_BLOCK_SIZE = 32;

void HaloDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    // validate input data tensor
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        // skip the untilize, only do halo
        log_debug(tt::LogOp, "Input is ROW_MAJOR, no need to untilize.");
    } else {
        TT_FATAL(input_tensor.physical_volume() % tt::constants::TILE_HW == 0, "Error");
    }
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Only height, width or block sharded tensors are supported.");
    TT_FATAL(input_tensor.shard_spec().has_value(), "Shard spec should not be empty");
}

std::vector<TensorSpec> HaloDeviceOperation::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.padded_shape();
    ttnn::Shape output_shape = ttnn::Shape(input_shape.to_array_4D());

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
    DataType output_dtype = (input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32)
                                ? tt::tt_metal::DataType::FLOAT32
                                : tt::tt_metal::DataType::BFLOAT16;

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == output_memory_config_.memory_layout(),
        "{} {}",
        input_tensor.memory_config(),
        output_memory_config_);

    if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        auto input_core_range = *(input_tensor.memory_config().shard_spec()->grid.ranges().begin());
        auto output_core_range = *(output_memory_config_.shard_spec()->grid.ranges().begin());
        auto input_core_w = input_core_range.end_coord.y - input_core_range.start_coord.y + 1;
        auto output_core_w = output_core_range.end_coord.y - output_core_range.start_coord.y + 1;
        TT_FATAL(input_core_w == output_core_w, "Error");
    }

    if (this->in_place_) {
        log_info(tt::LogOp, "halo_device_operation - Using in-place mode so deallocating input buffer");
        // TODO: `input_tensor` is const qualified, but Tensor::deallocate() is not.
        // Find a nicer way to do this.
        input_tensor.mesh_buffer()->deallocate();
    }

    std::array<uint32_t, 2> shard_shape = {
        tt::div_up(output_shape[0] * output_shape[2], config_.num_cores_nhw),
        input_tensor.memory_config().shard_spec()->shape[1]};
    auto out_mem_config = output_memory_config_.with_shard_spec(ShardSpec{
        output_memory_config_.shard_spec()->grid,
        shard_shape,
        shard_shape,
        output_memory_config_.shard_spec()->orientation});
    return {TensorSpec(output_shape, TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), out_mem_config))};
}

operation::ProgramWithCallbacks HaloDeviceOperation::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    const auto& input_tensor = inputs.at(0);
    auto& output_tensor = outputs.at(0);
    auto device = input_tensor.device();

    bool is_in_tiled = input_tensor.layout() == Layout::TILE;
    bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    auto pad_metadata = sliding_window::generate_pad_metadata(config_);
    auto op_trace_metadata = sliding_window::generate_op_trace_metadata(config_);
    auto shard_boundaries = sliding_window::generate_shard_boundaries(config_, op_trace_metadata);
    uint32_t input_shard_height = input_tensor.memory_config().shard_spec()->shape[0];
    auto tensor_metadata = sliding_window::generate_tensor_metadata(pad_metadata, config_, input_shard_height);

    Program program = CreateProgram();

    if (this->in_place_) {
        auto kernel_config = sliding_window::generate_inplace_halo_kernel_config_tensors(
            tensor_metadata,
            shard_boundaries,
            is_block_sharded,
            transpose_mcast_,
            remote_read_,
            is_in_tiled,
            device,
            max_out_nsticks_per_core_,
            in_nsticks_per_core_,
            this->in_place_);

        const auto& pad_config1 = std::get<0>(kernel_config)[0];
        const auto& local_config1 = std::get<0>(kernel_config)[2];
        const auto& remote_config1 = std::get<0>(kernel_config)[4];
        const auto& max_ref_size = std::get<1>(kernel_config);

        auto pad_config_tensor1 = sliding_window::construct_on_host_config_tensor(pad_config1, this->parallel_config_);
        auto local_config_tensor1 =
            sliding_window::construct_on_host_config_tensor(local_config1, this->parallel_config_);
        auto remote_config_tensor1 =
            sliding_window::construct_on_host_config_tensor(remote_config1, this->parallel_config_);

        auto pad_config_device_tensor1 = sliding_window::move_config_tensor_to_device(
            pad_config_tensor1, parallel_config_, is_block_sharded, device);
        auto local_config_device_tensor1 = sliding_window::move_config_tensor_to_device(
            local_config_tensor1, parallel_config_, is_block_sharded, device);
        auto remote_config_device_tensor1 = sliding_window::move_config_tensor_to_device(
            remote_config_tensor1, parallel_config_, is_block_sharded, device);

        DataType type = input_tensor.dtype();
        int num_cores = this->parallel_config_.grid.num_cores();
        int num_cores_c = conv::get_num_cores_channels_from_parallel_config(this->parallel_config_);
        int stick_size = input_tensor.padded_shape()[3] / num_cores_c;

        int pad_h = config_.get_pad_h() + config_.get_ceil_pad_h();
        int pad_w = config_.get_pad_w() + config_.get_ceil_pad_w();
        bool padding_exists = pad_h > 0 || pad_w > 0;

        return {data_movement::detail::inplace_untilize_with_halo_multi_core(
            program,
            input_tensor,
            pad_val_,
            padding_exists,
            config_.num_cores_nhw,
            config_.num_cores_c,
            max_out_nsticks_per_core_,
            max_ref_size,
            pad_config_device_tensor1,
            local_config_device_tensor1,
            remote_config_device_tensor1,
            remote_read_,
            transpose_mcast_,
            output_tensor,
            /*capture_buffers=*/true)};

    } else {
        auto kernel_config = sliding_window::generate_halo_kernel_config_tensors(
            tensor_metadata,
            shard_boundaries,
            is_block_sharded,
            transpose_mcast_,
            remote_read_,
            device,
            is_in_tiled,
            UNTILIZE_BLOCK_SIZE);

        const auto& pad_config = kernel_config.pad_config;
        const auto& gather_config0 = kernel_config.gather_config0;
        const auto& gather_config1 = kernel_config.gather_config1;

        const auto pad_config_tensor =
            sliding_window::construct_on_host_config_tensor(pad_config, this->parallel_config_);
        const auto gather_config_tensor0 =
            sliding_window::construct_on_host_config_tensor(gather_config0, this->parallel_config_);
        const auto gather_config_tensor1 =
            sliding_window::construct_on_host_config_tensor(gather_config1, this->parallel_config_);

        auto pad_config_device_tensor =
            sliding_window::move_config_tensor_to_device(pad_config_tensor, parallel_config_, is_block_sharded, device);
        auto gather_config_device_tensor0 = sliding_window::move_config_tensor_to_device(
            gather_config_tensor0, parallel_config_, is_block_sharded, device);
        auto gather_config_device_tensor1 = sliding_window::move_config_tensor_to_device(
            gather_config_tensor1, parallel_config_, is_block_sharded, device);

        const auto number_of_blocks_per_core = sliding_window::remap_nhw_scalar_argument_across_full_grid(
            kernel_config.number_of_blocks_per_core, this->parallel_config_);

        Program program = CreateProgram();

        return {data_movement::detail::untilize_with_halo_multi_core(
            program,
            input_tensor,
            pad_val_,
            config_.num_cores_nhw,
            max_out_nsticks_per_core_,
            pad_config_device_tensor,
            gather_config_device_tensor0,
            gather_config_device_tensor1,
            number_of_blocks_per_core,
            remote_read_,
            transpose_mcast_,
            output_tensor,
            UNTILIZE_BLOCK_SIZE,
            /*capture_buffers=*/true)};
    }
}

Tensor halo_op(
    const Tensor& input_tensor,
    const SlidingWindowConfig& config,
    uint32_t pad_val,
    bool remote_read,
    bool transpose_mcast,
    const MemoryConfig& output_memory_config,
    bool is_out_tiled,
    bool in_place) {
    TT_FATAL(input_tensor.memory_config().is_sharded(), "Halo expects sharded input tensor");
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Only height, width or block sharded tensors are supported.");
    // NOTE: for HEIGHT_SHARDED, ncores_nhw == ncores
    //       for BLOCK_SHARDED, ncores_nhw is just the ncores along height dim (last tensor dim is split along
    //       width)
    bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

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
    uint32_t in_nsticks_per_core = input_tensor.memory_config().shard_spec()->shape[0];
    ParallelConfig p_config;
    p_config.grid = input_tensor.shard_spec().value().grid;
    p_config.shard_scheme = input_tensor.memory_config().memory_layout();
    p_config.shard_orientation = input_tensor.shard_spec().value().orientation;

    if (in_place && in_nsticks_per_core > max_out_nsticks_per_core) {
        log_info(
            tt::LogOp,
            "halo_device_operation - in place operation is not supported for parameterizations with "
            "input shard size larger than output shard size, falling back to normal operation");
        in_place = false;
    }

    return operation::run(
               HaloDeviceOperation{
                   .config_ = config,
                   .parallel_config_ = p_config,
                   .pad_val_ = pad_val,
                   .remote_read_ = remote_read,
                   .transpose_mcast_ = transpose_mcast,
                   .max_out_nsticks_per_core_ = max_out_nsticks_per_core,
                   .in_nsticks_per_core_ = in_nsticks_per_core,
                   .output_memory_config_ = output_memory_config,
                   .is_out_tiled_ = is_out_tiled,
                   .in_place_ = in_place},
               {input_tensor})
        .at(0);
}

}  // namespace ttnn::operations::sliding_window::halo
