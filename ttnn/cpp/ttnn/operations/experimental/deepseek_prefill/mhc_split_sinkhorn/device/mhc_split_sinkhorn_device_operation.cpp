// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_split_sinkhorn_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

void MhcSplitSinkhornDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& mixes = tensor_args.mixes;
    const auto& consts = tensor_args.consts;
    TT_FATAL(mixes.dtype() == tt::tt_metal::DataType::FLOAT32, "mixes must be FLOAT32");
    TT_FATAL(consts.dtype() == tt::tt_metal::DataType::FLOAT32, "consts must be FLOAT32");
    TT_FATAL(mixes.layout() == tt::tt_metal::Layout::TILE, "mixes must be TILE layout");
    TT_FATAL(consts.layout() == tt::tt_metal::Layout::TILE, "consts must be TILE layout");
    const uint32_t mix_hc = (2 + args.n) * args.n;
    // >= so a sharded mixes may be padded up to a tile-aligned width (32); the kernel reads
    // only the first mix_hc columns via the SEL matmul.
    TT_FATAL(
        mixes.logical_shape()[-1] >= mix_hc,
        "mixes last dim ({}) must be >= (2+n)*n={}",
        mixes.logical_shape()[-1],
        mix_hc);
    TT_FATAL(args.n * args.n <= 32, "n*n ({}) must fit one tile width (32)", args.n * args.n);
    TT_FATAL(consts.physical_volume() / tt::constants::TILE_HW == 8, "consts must be 8 tiles [8,32,32]");
}

MhcSplitSinkhornDeviceOperation::spec_return_value_t MhcSplitSinkhornDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& mixes = tensor_args.mixes;
    const uint32_t T = mixes.logical_shape()[-2];
    const uint32_t n = args.n;

    // A sharded input yields sharded outputs on the same grid (same tokens-per-core), so
    // each core writes its result straight to its L1 shard; otherwise DRAM interleaved.
    auto spec_for = [&](uint32_t w) {
        if (mixes.memory_config().is_sharded()) {
            // Sharded TILE tensors require a tile-aligned shard width, so sharded outputs are a
            // full 32 wide (the kernel already emits 32-wide tiles); the caller slices [:, :w].
            const auto& in_shard = mixes.shard_spec().value();
            const uint32_t tw = tt::constants::TILE_WIDTH;
            tt::tt_metal::ShardSpec out_shard(in_shard.grid, {in_shard.shape[0], tw}, in_shard.orientation);
            auto mem = tt::tt_metal::MemoryConfig(
                mixes.memory_config().memory_layout(), tt::tt_metal::BufferType::L1, out_shard);
            return TensorSpec(
                ttnn::Shape({T, tw}),
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::FLOAT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem));
        }
        return TensorSpec(
            ttnn::Shape({T, w}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig{
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
    };
    return {spec_for(n), spec_for(n), spec_for(n * n)};
}

MhcSplitSinkhornDeviceOperation::tensor_return_value_t MhcSplitSinkhornDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.mixes.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
        create_device_tensor(specs[2], device)};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::array<Tensor, 3> mhc_split_sinkhorn(
    const Tensor& mixes, const Tensor& consts, uint32_t n, uint32_t sinkhorn_iters, float eps) {
    using OperationType = ttnn::experimental::prim::MhcSplitSinkhornDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{n, sinkhorn_iters, eps};
    auto tensor_args = OperationType::tensor_args_t{mixes, consts};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
