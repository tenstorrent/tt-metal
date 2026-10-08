// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ag_row_ops_device_operation.hpp"

#include "moe_ag_common.hpp"

using namespace tt::tt_metal;
using namespace ttnn::operations::experimental::deepseek_prefill::moe_ag;

namespace ttnn::prim {

// Blocks (32 rows x 1024 columns) j = me, me + P, ... per core; num_blocks == 2: the a + b add (c_0 / c_1 -> c_24
// row major -> tilize to c_16), else N blocks accumulated by the packer in L1.
ProgramDescriptor MoeAgSumRowsTiledDeviceOperation::ProgramFactory::create_descriptor(
    const MoeAgSumRowsTiledParams& args, const MoeAgSumRowsTiledInputs& t, Tensor& out) {
    const uint32_t H = t.src.logical_shape()[-1], NCH = H / 1024;
    const auto grid = worker_grid(t.src);
    const auto& crs = grid.range;
    const uint32_t P = grid.cores, gx = grid.size.x;
    const uint32_t blocks = args.num_rows / 32 * NCH;

    ProgramDescriptor desc;
    KernelDescriptor reader, compute;
    if (args.num_blocks == 2) {
        desc.cbs = {tile_cb(0, 64, crs), tile_cb(1, 64, crs), tile_cb(24, 32, crs), tile_cb(16, 64, crs)};
        reader = kernel_desc("addt_reader.cpp", crs, {H * 2, NCH, P}, dm_config(1, 1));
        reader.emplace_common_runtime_args({t.src.buffer(), t.src.buffer(), 0u, args.block_stride, blocks, gx});
        compute = kernel_desc("addt_compute.cpp", crs, {}, fp32_compute_config());
    } else {
        desc.cbs = {tile_cb(0, 64, crs), tile_cb(24, 32, crs), tile_cb(16, 64, crs)};
        reader = kernel_desc("addn_reader.cpp", crs, {H * 2, NCH, P, args.num_blocks}, dm_config(1, 1));
        reader.emplace_common_runtime_args({t.src.buffer(), args.block_stride, blocks, gx});
        compute = kernel_desc("addn_compute.cpp", crs, {args.num_blocks}, fp32_compute_config());
    }
    compute.emplace_common_runtime_args({blocks, P, gx});
    auto writer = kernel_desc("addt_writer.cpp", crs, {NCH, P, H / 32}, dm_config(0, 0));
    writer.emplace_common_runtime_args({out.buffer(), blocks, gx});
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    return desc;
}

// Rows [r0, r0 + n) per core (core_range, no alignment), `batch` rows per read barrier.
ProgramDescriptor MoeAgAddRowsDeviceOperation::ProgramFactory::create_descriptor(
    const MoeAgAddRowsParams& args, const MoeAgAddRowsInputs& t, Tensor& out) {
    const uint32_t H = t.a.logical_shape()[-1], RB = H * 2, TILES = H / 1024, B = args.batch;
    const auto grid = worker_grid(t.a);
    const auto& crs = grid.range;
    const uint32_t gx = grid.size.x, n = args.num_rows, per = per_core(n, grid.cores, 1);

    ProgramDescriptor desc;
    desc.cbs = {
        tile_cb(0, 2 * B * TILES, crs),
        tile_cb(1, 2 * B * TILES, crs),
        scratch_cb(7, 64, crs),
        tile_cb(16, 2 * TILES, crs),
    };
    auto reader = kernel_desc("add_reader.cpp", crs, {RB, TILES, uint32_t(args.info_offset), B}, dm_config(1, 1));
    reader.emplace_common_runtime_args(
        {t.a.buffer(), t.b.buffer(), t.chip_info.buffer(), n, per, args.b_offset, args.a_offset, gx});
    auto compute = kernel_desc("add_compute.cpp", crs, {TILES}, fp32_compute_config());
    compute.emplace_common_runtime_args({n, per, gx});
    auto writer = kernel_desc("reduce_writer.cpp", crs, {RB, TILES, 1, 0}, dm_config(0, 0));
    writer.emplace_common_runtime_args({out.buffer(), 0u, 0u, n, per, gx});
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    return desc;
}

// Active tile rows dealt round robin (block j = (expert tile row, W-tile chunk) on core j % P), bfp8 tiles (1088 B).
ProgramDescriptor MoeAgUntilizeActiveDeviceOperation::ProgramFactory::create_descriptor(
    const MoeAgUntilizeActiveParams& args, const MoeAgUntilizeActiveInputs& t, Tensor& out) {
    const uint32_t H = t.y.logical_shape()[-1], W = args.tiles_per_block, NCH = H / (32 * W);
    const uint32_t NG = t.local_slot_map.logical_shape()[-1], EPC = args.experts_per_chip;
    constexpr uint32_t BFP8_TILE = 1088;
    const auto grid = worker_grid(t.y);
    const auto& crs = grid.range;
    const uint32_t P = grid.cores, gx = grid.size.x;

    ProgramDescriptor desc;
    desc.cbs = {
        tile_cb(0, 2 * W, crs, tt::DataFormat::Bfp8_b, BFP8_TILE),
        scratch_cb(2, 64, crs),
        scratch_cb(4, 3 * NG * 4, crs),
        scratch_cb(5, 3 * NG * 4, crs),
        tile_cb(16, 2 * W, crs),
    };
    auto reader = kernel_desc("untilize_reader.cpp", crs, {NG, EPC, BFP8_TILE, W, NCH, P}, dm_config(1, 1));
    reader.emplace_common_runtime_args(
        {t.y.buffer(), t.counts.buffer(), t.regions.buffer(), t.local_slot_map.buffer(), gx});
    auto compute = kernel_desc("untilize_compute.cpp", crs, {W}, ComputeConfigDescriptor{});
    auto writer = kernel_desc("untilize_writer.cpp", crs, {NG, EPC, W, NCH, P}, dm_config(0, 0));
    writer.emplace_common_runtime_args(
        {out.buffer(), t.counts.buffer(), t.regions.buffer(), t.local_slot_map.buffer(), gx});
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    return desc;
}

// Blocks (32 rows x 1024 columns) j = me, me + P, ... per core: 32 tiles untilized into 32 2 KB pages.
ProgramDescriptor MoeAgUntilizeXDeviceOperation::ProgramFactory::create_descriptor(
    const MoeAgUntilizeXParams&, const MoeAgUntilizeXInputs& t, Tensor& out) {
    const uint32_t H = t.x.logical_shape()[-1], NCH = H / 1024;
    const uint32_t rows = t.x.physical_volume() / H;
    const auto grid = worker_grid(t.x);
    const auto& crs = grid.range;
    const uint32_t P = grid.cores, gx = grid.size.x, blocks = rows / 32 * NCH;

    ProgramDescriptor desc;
    desc.cbs = {tile_cb(0, 64, crs), scratch_cb(2, 64, crs), tile_cb(16, 64, crs)};
    auto reader = kernel_desc("untilize_x_reader.cpp", crs, {2048, NCH, P}, dm_config(1, 1));
    reader.emplace_common_runtime_args({t.x.buffer(), blocks, gx});
    auto compute = kernel_desc("untilize_compute.cpp", crs, {32}, ComputeConfigDescriptor{});
    auto writer = kernel_desc("untilize_x_writer.cpp", crs, {NCH, P}, dm_config(0, 0));
    writer.emplace_common_runtime_args({out.buffer(), blocks, gx});
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    return desc;
}

}  // namespace ttnn::prim
