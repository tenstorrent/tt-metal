// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_halo_program_factory.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/sliding_window/halo/device/halo_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/mesh_buffer.hpp>  // complete MeshBuffer type for the in-place input deallocation
#include <array>

namespace ttnn::prim {

using namespace tt::tt_metal;

thread_local std::unordered_map<std::size_t, std::uint32_t>
    HaloDeviceOperation::sliding_window_max_out_nsticks_per_core = {};

// TODO: Look into increasing this to tradeoff some L1 for performance (#19980)

void HaloDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args;

    // validate input data tensor
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        // skip the untilize, only do halo
        log_debug(tt::LogOp, "Input is ROW_MAJOR, no need to untilize.");
    } else {
        TT_FATAL(
            input_tensor.physical_volume() % tt::constants::TILE_HW == 0,
            "Input tensor physical volume ({}) must be divisible by TILE_HW ({})",
            input_tensor.physical_volume(),
            tt::constants::TILE_HW);
    }
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Only height, width or block sharded tensors are supported.");
    TT_FATAL(input_tensor.shard_spec().has_value(), "Shard spec should not be empty");
}

HaloDeviceOperation::spec_return_value_t HaloDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args;
    const auto& input_shape = input_tensor.padded_shape();
    ttnn::Shape output_shape = ttnn::Shape(input_shape.to_array_4D());

    uint32_t nbatch = input_shape[0];
    uint32_t total_nsticks = args.config.num_cores_nhw * args.max_out_nsticks_per_core;

    // output_shape[0] remains same
    // output_shape[1] remains same
    // output_shape[2] changes
    // output_shape[3] remains same
    output_shape[2] = (uint32_t)std::ceil((float)total_nsticks / nbatch);

    log_debug(
        tt::LogOp, "output_shape: [{} {} {} {}]", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    log_debug(tt::LogOp, "max_out_nsticks_per_core: {}", args.max_out_nsticks_per_core);
    log_debug(
        tt::LogOp, "size : {}", args.in_nsticks_per_core * input_tensor.memory_config().shard_spec()->shape[1] * 2);
    log_debug(tt::LogOp, "num_cores_nhw: {}", args.config.num_cores_nhw);

    tt::tt_metal::DataType output_dtype;
    switch (input_tensor.dtype()) {
        case tt::tt_metal::DataType::FLOAT32: output_dtype = tt::tt_metal::DataType::FLOAT32; break;
        case tt::tt_metal::DataType::UINT16: output_dtype = tt::tt_metal::DataType::UINT16; break;
        default: output_dtype = tt::tt_metal::DataType::BFLOAT16; break;
    }

    std::array<uint32_t, 2> shard_shape = {
        tt::div_up(output_shape[0] * output_shape[2], args.config.num_cores_nhw),
        input_tensor.memory_config().shard_spec()->shape[1]};

    auto out_mem_config = MemoryConfig(
        input_tensor.memory_config().memory_layout(),
        input_tensor.memory_config().buffer_type(),
        ShardSpec{
            input_tensor.memory_config().shard_spec()->grid,
            shard_shape,
            input_tensor.memory_config().shard_spec()->orientation});
    auto padded_output_shape = output_shape;
    padded_output_shape[-2] = tt::round_up(padded_output_shape[-2], shard_shape[0]);
    padded_output_shape[-1] = tt::round_up(padded_output_shape[-1], shard_shape[1]);
    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            output_dtype, PageConfig(Layout::ROW_MAJOR), out_mem_config, output_shape, padded_output_shape));
}

HaloDeviceOperation::tensor_return_value_t HaloDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(args, tensor_args);
    const auto& input_tensor = tensor_args;

    // In-place halo (silent auto-activation; see IN_PLACE_HALO_REDO.md sec 10). The SAME pure
    // decision function is used by the program factory and the pool caller so all three agree.
    const bool is_in_tiled = input_tensor.layout() == Layout::TILE;
    const bool in_place = ttnn::operations::sliding_window::should_halo_be_in_place(
        args.allow_in_place,
        args.config,
        args.in_nsticks_per_core,
        input_tensor.memory_config().memory_layout(),
        is_in_tiled);
    if (!in_place) {
        return create_device_tensor(output_spec, input_tensor.device());
    }
    log_info(tt::LogOp, "halo_device_operation - in-place halo active; aliasing output onto input buffer");

    // Capture the input shard address AND its aligned per-bank size BEFORE freeing it (both used
    // to assert the overlap below). For TILED input the aliasing offset is the buffer-size delta
    // (untilize writes into the dst buffer, so the stick-index delta is 0 and cannot be used) --
    // that needs the input buffer's byte size before it is deallocated.
    Buffer* src_buffer = input_tensor.buffer();
    const uint32_t src_addr = src_buffer->address();
    const uint32_t in_aligned_size_per_bank = src_buffer->aligned_size_per_bank();

    // Aliasing delta between input and output shards (MUST match the factory build_inplace_halo_program).
    //   Row-major: stick-based delta = (aligned out_nsticks - aligned in_nsticks) * post-untilize width.
    //   Tiled:     buffer-size delta = out_buffer.aligned_size_per_bank() - in_buffer.aligned_size_per_bank(),
    //              because the tiled input has a different byte layout than the untilized row-major output
    //              and its tiles physically live at the TAIL of the output buffer (#27333/#30644).
    // The tiled buffer_delta is finalized after the output buffer is allocated (needs dst size).
    uint32_t delta_bytes = 0;
    if (!is_in_tiled) {
        const uint32_t shard_width = input_tensor.memory_config().shard_spec()->shape[1];
        const uint32_t nbytes = input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ? 4u : 2u;
        const uint32_t width_bytes = shard_width * nbytes;
        const uint32_t aligned_delta_size =
            (ttnn::operations::sliding_window::align_buffer(args.max_out_nsticks_per_core * width_bytes) /
             width_bytes) -
            (ttnn::operations::sliding_window::align_buffer(args.in_nsticks_per_core * width_bytes) / width_bytes);
        delta_bytes = aligned_delta_size * width_bytes;
    }

    // Free the input shard's L1 region so the (larger) output shard can be allocated over the
    // same top-of-L1 region (sharded L1 allocates top-down). Deallocating the underlying
    // MeshBuffer (not the Tensor) frees the owning allocation while keeping the input Tensor's
    // storage/buffer() queryable -- the framework's collect_tensor_buffers reads input.buffer()
    // during workload build. The input aliases into the output after this, so the pool/conv
    // caller MUST skip its own input-dealloc when in-place is active.
    // TODO: mesh_buffer() is const-qualified but MeshBuffer::deallocate() is not.
    const_cast<tt::tt_metal::distributed::MeshBuffer&>(input_tensor.mesh_buffer()).deallocate();

    auto output = create_device_tensor(output_spec, input_tensor.device());
    Buffer* dst_buffer = output.buffer();
    if (is_in_tiled) {
        delta_bytes = dst_buffer->aligned_size_per_bank() - in_aligned_size_per_bank;
    }
    TT_FATAL(
        src_addr == dst_buffer->address() + delta_bytes,
        "In-place halo requires the input shard buffer to overlap the output shard buffer at the expected "
        "offset (src {} != dst {} + delta {})",
        src_addr,
        dst_buffer->address(),
        delta_bytes);
    return output;
}

Tensor halo(
    const Tensor& input_tensor,
    const ttnn::operations::sliding_window::SlidingWindowConfig& config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t pad_val,
    bool remote_read,
    bool transpose_mcast,
    bool is_out_tiled,
    bool config_tensors_in_dram,
    bool allow_in_place) {
    using OperationType = HaloDeviceOperation;

    TT_FATAL(input_tensor.memory_config().is_sharded(), "Halo expects sharded input tensor");
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Only height, width or block sharded tensors are supported.");
    // NOTE: for HEIGHT_SHARDED, ncores_nhw == ncores
    //       for BLOCK_SHARDED, ncores_nhw is just the ncores along height dim (last tensor dim is split along
    //       width)
    auto sliding_window_hash = config.get_hash();
    if (!OperationType::sliding_window_max_out_nsticks_per_core.contains(sliding_window_hash)) {
        auto op_trace_metadata = ttnn::operations::sliding_window::generate_op_trace_metadata(config);
        auto shard_boundaries = ttnn::operations::sliding_window::generate_shard_boundaries(config);
        OperationType::sliding_window_max_out_nsticks_per_core.emplace(
            sliding_window_hash, ttnn::operations::sliding_window::generate_max_out_nsticks_per_core(shard_boundaries));
    }

    uint32_t max_out_nsticks_per_core = OperationType::sliding_window_max_out_nsticks_per_core.at(sliding_window_hash);
    uint32_t in_nsticks_per_core = input_tensor.memory_config().shard_spec()->shape[0];
    ttnn::operations::sliding_window::ParallelConfig p_config;
    p_config.grid = input_tensor.shard_spec().value().grid;
    p_config.shard_scheme = input_tensor.memory_config().memory_layout();
    p_config.shard_orientation = input_tensor.shard_spec().value().orientation;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .config = config,
            .parallel_config = p_config,
            .pad_val = pad_val,
            .remote_read = remote_read,
            .transpose_mcast = transpose_mcast,
            .max_out_nsticks_per_core = max_out_nsticks_per_core,
            .in_nsticks_per_core = in_nsticks_per_core,
            .is_out_tiled = is_out_tiled,
            .config_tensors_in_dram = config_tensors_in_dram,
            .allow_in_place = allow_in_place,
            .compute_kernel_config = compute_kernel_config},
        input_tensor);
}
}  // namespace ttnn::prim
