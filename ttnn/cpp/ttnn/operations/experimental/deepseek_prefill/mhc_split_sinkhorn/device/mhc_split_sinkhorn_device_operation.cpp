// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_split_sinkhorn_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <cstdlib>

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

    TT_FATAL(mixes.is_allocated() && consts.is_allocated(), "mixes and consts must be allocated on device");
    TT_FATAL(
        mixes.storage_type() == StorageType::DEVICE && consts.storage_type() == StorageType::DEVICE,
        "mixes and consts must be device tensors");
    TT_FATAL(mixes.device() == consts.device(), "mixes and consts must be on the same device");
    // T is read from logical_shape[-2] while token-tiles come from physical_volume/TILE_HW, so the
    // input must be an effectively 2D [T, w] block at most one tile wide -- otherwise the writer
    // would index pages beyond the output buffers.
    const auto& ls = mixes.logical_shape();
    TT_FATAL(ls.rank() >= 2, "mixes must be rank>=2 ([T, w]), got rank {}", ls.rank());
    for (int i = 0; i + 2 < static_cast<int>(ls.rank()); ++i) {
        TT_FATAL(ls[i] == 1, "mixes leading dim {} must be 1 (got {})", i, ls[i]);
    }
    TT_FATAL(
        mixes.padded_shape()[-1] <= tt::constants::TILE_WIDTH,
        "mixes must fit one tile wide (padded width {} > {})",
        mixes.padded_shape()[-1],
        tt::constants::TILE_WIDTH);
    if (mixes.memory_config().is_sharded()) {
        TT_FATAL(
            mixes.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            "sharded mixes must be HEIGHT-sharded (each core owns whole one-tile-wide token rows)");
        TT_FATAL(
            mixes.shard_spec().value().shape[1] == tt::constants::TILE_WIDTH,
            "sharded mixes shard width ({}) must be one tile ({})",
            mixes.shard_spec().value().shape[1],
            tt::constants::TILE_WIDTH);
    }
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
            return tt::tt_metal::TensorSpec(
                ttnn::Shape({T, tw}),
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::FLOAT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem));
        }
        return tt::tt_metal::TensorSpec(
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
    // MHC_MAX_CORES=1 pins to a single core (the multi-core A/B baseline). Read it here and carry
    // it in the hashed attributes so a changed value re-keys the program cache rather than
    // silently reusing a program built for the other core grid.
    uint32_t max_cores = 0;
    if (const char* mc = std::getenv("MHC_MAX_CORES"); mc != nullptr) {
        max_cores = static_cast<uint32_t>(std::atoi(mc));
    }
    auto operation_attributes = OperationType::operation_attributes_t{n, sinkhorn_iters, eps, max_cores};
    auto tensor_args = OperationType::tensor_args_t{mixes, consts};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
