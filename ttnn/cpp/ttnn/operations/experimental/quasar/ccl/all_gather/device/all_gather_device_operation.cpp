// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_device_operation.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;

AllGatherDeviceOperation::program_factory_t AllGatherDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return AllGatherProgramFactory{};
}

void AllGatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input_tensor;
    validate_on_program_cache_hit(args, tensor_args);

    TT_FATAL(input.layout() == Layout::TILE, "quasar.all_gather: input must be TILE layout, got {}", input.layout());
    TT_FATAL(
        input.memory_config().is_sharded() && input.shard_spec().has_value(),
        "quasar.all_gather: input must be height- or width-sharded (one shard per rank), got {}",
        input.memory_config().memory_layout());
    const auto layout = input.memory_config().memory_layout();
    TT_FATAL(
        layout == TensorMemoryLayout::HEIGHT_SHARDED || layout == TensorMemoryLayout::WIDTH_SHARDED,
        "quasar.all_gather: input must be HEIGHT_SHARDED or WIDTH_SHARDED, got {}",
        layout);
    TT_FATAL(
        input.dtype() != DataType::BFLOAT8_B && input.dtype() != DataType::BFLOAT4_B,
        "quasar.all_gather: Quasar has no Bfp8/Bfp4; got {}",
        input.dtype());

    const auto& shard = input.shard_spec().value();
    const uint32_t num_ranks = shard.grid.num_cores();
    TT_FATAL(num_ranks >= 2, "quasar.all_gather: need at least 2 ranks (shards), got {}", num_ranks);

    const auto& padded = input.padded_shape();
    const uint32_t height = padded.volume() / padded[-1];
    const uint32_t width = padded[-1];
    if (layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_FATAL(
            shard.shape[0] * num_ranks == height && shard.shape[1] == width,
            "quasar.all_gather: height shards must tile the tensor exactly: shard {}x{} x {} ranks vs tensor {}x{}",
            shard.shape[0],
            shard.shape[1],
            num_ranks,
            height,
            width);
    } else {
        TT_FATAL(
            shard.shape[1] * num_ranks == width && shard.shape[0] == height,
            "quasar.all_gather: width shards must tile the tensor exactly: shard {}x{} x {} ranks vs tensor {}x{}",
            shard.shape[0],
            shard.shape[1],
            num_ranks,
            height,
            width);
    }
    TT_FATAL(
        shard.shape[0] % tt::constants::TILE_HEIGHT == 0 && shard.shape[1] % tt::constants::TILE_WIDTH == 0,
        "quasar.all_gather: shard shape {}x{} must be tile aligned",
        shard.shape[0],
        shard.shape[1]);

    TT_FATAL(args.tiles_per_packet >= 1, "quasar.all_gather: tiles_per_packet must be >= 1");
    TT_FATAL(args.chunks_per_sync >= 1, "quasar.all_gather: chunks_per_sync must be >= 1");
    TT_FATAL(args.num_workers_per_rank >= 1, "quasar.all_gather: num_workers_per_rank must be >= 1");
    if (args.topology == AllGatherTopology::DIRECT) {
        TT_FATAL(
            args.reader_threads >= 1 && args.reader_threads <= 6,
            "quasar.all_gather(direct): reader_threads must be in [1, 6] (6 user DM cores per cluster), got {}",
            args.reader_threads);
        TT_FATAL(
            args.num_workers_per_rank == 1,
            "quasar.all_gather(direct): reads land in the home cluster's own shard, so num_workers_per_rank must be "
            "1, got {}",
            args.num_workers_per_rank);
    } else {
        TT_FATAL(
            args.reader_threads == args.writer_threads,
            "quasar.all_gather(ring): reader_threads ({}) must equal writer_threads ({}) so writer thread t drains "
            "exactly what reader thread t produced",
            args.reader_threads,
            args.writer_threads);
        TT_FATAL(
            args.reader_threads >= 1 && args.reader_threads + args.writer_threads <= 6,
            "quasar.all_gather(ring): reader_threads + writer_threads must be in [2, 6] (6 user DM cores per "
            "cluster), got {}",
            args.reader_threads + args.writer_threads);
    }
}

void AllGatherDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input_tensor;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "quasar.all_gather: input must be on device");
    TT_FATAL(input.buffer() != nullptr, "quasar.all_gather: input buffer must be allocated");
}

AllGatherDeviceOperation::spec_return_value_t AllGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input_tensor;
    const auto& shard = input.shard_spec().value();
    const uint32_t num_ranks = shard.grid.num_cores();

    // [N, ...input shape...]; every output shard is the whole gathered tensor.
    const auto& in_shape = input.logical_shape();
    ttsl::SmallVector<uint32_t> dims;
    dims.reserve(in_shape.rank() + 1);
    dims.push_back(num_ranks);
    for (uint32_t i = 0; i < in_shape.rank(); i++) {
        dims.push_back(in_shape[i]);
    }
    const ttnn::Shape out_shape(dims);

    const auto& padded = input.padded_shape();
    const uint32_t height = padded.volume() / padded[-1];
    const uint32_t width = padded[-1];
    const ShardSpec out_shard(shard.grid, {height, width}, shard.orientation);
    const MemoryConfig out_mem(TensorMemoryLayout::HEIGHT_SHARDED, input.memory_config().buffer_type(), out_shard);
    return TensorSpec(out_shape, TensorLayout(input.dtype(), PageConfig(input.layout()), out_mem));
}

AllGatherDeviceOperation::tensor_return_value_t AllGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

Tensor all_gather(const Tensor& input_tensor, const AllGatherOperationAttributes& attributes) {
    return ttnn::device_operation::launch<AllGatherDeviceOperation>(
        attributes, AllGatherTensorArgs{.input_tensor = input_tensor});
}

}  // namespace ttnn::prim::qsr
