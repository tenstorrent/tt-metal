// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pull_all_gather_device_operation.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::ccl {

PullAllGatherDeviceOperation::spec_return_value_t PullAllGatherDeviceOperation::compute_output_specs(
    const PullAllGatherParams& args, const PullAllGatherInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto shape = input_tensor.logical_shape();
    shape[args.dim] *= args.num_devices;  // (1,1,M,N) -> (1,1,M*num_devices,N)
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), args.output_mem_config));
}

PullAllGatherDeviceOperation::tensor_return_value_t PullAllGatherDeviceOperation::create_output_tensors(
    const PullAllGatherParams& args, const PullAllGatherInputs& tensor_args) {
    if (tensor_args.persistent_output_tensor.has_value()) {
        return tensor_args.persistent_output_tensor.value();
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

void PullAllGatherDeviceOperation::validate_on_program_cache_miss(
    const PullAllGatherParams& args, const PullAllGatherInputs& tensor_args) {
    const auto& input_spec = tensor_args.input_tensor.tensor_spec();
    const auto& shape = input_spec.logical_shape();

    TT_FATAL(
        args.dim == static_cast<int32_t>(shape.rank()) - 2, "This pull all-gather gathers on the row dim (-2) only");
    TT_FATAL(input_spec.layout() == tt::tt_metal::Layout::TILE, "TILE layout required");

    // Sharded so that one DMA descriptor covers one entry: a descriptor is
    // (src, dest + BAR, size), one contiguous run.
    TT_FATAL(
        args.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1 &&
            args.output_mem_config.nd_shard_spec().has_value(),
        "Output must be L1 sharded");

    // The input is read by the implicit-TRID overload, which fetches
    // get_entry_size() bytes from one address, so its pages must be contiguous
    // across an entry -- same requirement the output has, for the same reason.
    TT_FATAL(
        input_spec.memory_config().buffer_type() == tt::tt_metal::BufferType::L1 &&
            input_spec.memory_config().nd_shard_spec().has_value(),
        "Input must be L1 sharded; interleaved would put the next page in another bank");

    // Both height sharded: a shard's pages are one contiguous address run only
    // if the shard spans the full width. Nothing further is required of the two
    // shard specs -- they need not match, divide the block, or divide each
    // other, because the chunk rule takes the minimum of both remaining runs.
    // A device's last shard is allowed to be ragged.
    const auto& in_shard_shape = input_spec.memory_config().nd_shard_spec()->shard_shape;
    const auto& out_shard_shape = args.output_mem_config.nd_shard_spec()->shard_shape;
    const uint32_t tile_h = input_spec.tile().get_height();
    TT_FATAL(in_shard_shape.rank() >= 2 && out_shard_shape.rank() >= 2, "Shard shapes must have rank >= 2");
    TT_FATAL(
        in_shard_shape[-1] == shape[-1] && out_shard_shape[-1] == shape[-1],
        "Both must be height sharded: shard width must span the full N");
    TT_FATAL(
        in_shard_shape[-2] % tile_h == 0 && out_shard_shape[-2] % tile_h == 0, "Shard heights must be tile aligned");
    // The one alignment that matters: a device block is a whole number of
    // output shards, so the chunk walk is the same on every device and the
    // producer needs no device_idx. The input shard height is unconstrained --
    // its last shard per device may be ragged.
    TT_FATAL(
        (shape[-2] / tile_h) % (out_shard_shape[-2] / tile_h) == 0,
        "Output shard tile-row count must divide the per-device tile-row count");

    TT_FATAL(args.dfb_depth % args.num_producers == 0, "STRIDED producers must divide the DFB ring");

    // No topology restriction: the per-direction routes cover Ring and Linear
    // alike, including a Linear interior device, and a mesh.
}

PullAllGatherDeviceOperation::program_factory_t PullAllGatherDeviceOperation::select_program_factory(
    const PullAllGatherParams&, const PullAllGatherInputs&) {
    return PullAllGatherFactory{};  // one path: multicast pull
}

}  // namespace ttnn::operations::ccl
