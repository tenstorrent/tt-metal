// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ag_local_reduce_device_operation.hpp"

#include "moe_ag_common.hpp"

using namespace tt::tt_metal;
using namespace ttnn::operations::experimental::deepseek_prefill::moe_ag;

namespace ttnn::prim {

// Every worker core takes a token range (core_range over the tokens, 32-aligned when tiled). Reader (NCRISC, NOC1):
// per token a header page (its pair count) and the (y row, weight) pairs; compute: w * y accumulated by the packer
// in L1; writer (BRISC, NOC0): the reduced rows (or, tiled, tilized 32-row blocks).
// CBs: 0 y rows (D pairs), 1 weight tiles, 2 headers, 4 y_slot block, 5 weights, 6 zero row, 7 chip info,
// 16 reduced rows (tiled: 32 rows), 24 tiled: the row-major block before the tilize.
ProgramDescriptor MoeAgLocalReduceDeviceOperation::ProgramFactory::create_descriptor(
    const MoeAgLocalReduceParams& args, const MoeAgLocalReduceInputs& t, std::vector<Tensor>& outputs) {
    const uint32_t H = t.y.logical_shape()[-1], K = t.weights.logical_shape()[-1], T = rm_rows(t.weights);
    const uint32_t S = args.chunk_size_per_chip, D = args.pairs_depth;
    const uint32_t RB = H * 2, TILES = H / 1024;
    const bool tiled = args.phase != 1 && args.tiled;  // phase 0 (not split) or phase 2: tilized 32-row blocks
    const auto grid = worker_grid(t.y);
    const auto& crs = grid.range;
    const uint32_t gx = grid.size.x;
    const uint32_t total = args.phase == 0 ? T : S;
    const uint32_t per = per_core(total, grid.cores, tiled ? 32 : token_align(K));
    const uint32_t npr = std::min(per, total);

    ProgramDescriptor desc;
    desc.cbs.push_back(tile_cb(0, D * TILES, crs));
    desc.cbs.push_back(tile_cb(1, D, crs));
    desc.cbs.push_back(cb_desc(2, 128, 64, tt::DataFormat::UInt32, crs));
    desc.cbs.push_back(scratch_cb(4, round_up(npr * K * 4, 64), crs));
    desc.cbs.push_back(scratch_cb(5, round_up(npr * 64, 64), crs));
    desc.cbs.push_back(scratch_cb(6, RB, crs));
    desc.cbs.push_back(scratch_cb(7, 64, crs));
    desc.cbs.push_back(tile_cb(16, tiled ? 32 * TILES : 2 * TILES, crs));
    if (tiled) {
        desc.cbs.push_back(tile_cb(24, 32 * TILES, crs));
    }

    KernelDescriptor::RTArgList rng;  // (total, per, grid x): each core's range, kernels/core_range.hpp
    rng.push_back(total);
    rng.push_back(per);
    rng.push_back(gx);

    if (args.phase == 0) {
        auto reader = kernel_desc("reduce_reader.cpp", crs, {K, RB, TILES, 64}, dm_config(1, 1));
        reader.emplace_common_runtime_args({t.y.buffer(), t.y_slot.buffer(), t.weights.buffer(), total, per, gx});
        auto compute =
            kernel_desc(tiled ? "reduce_compute_t.cpp" : "reduce_compute.cpp", crs, {TILES}, fp32_compute_config());
        compute.emplace_common_runtime_args(rng);
        KernelDescriptor writer;
        if (tiled) {
            writer = kernel_desc("reduce_writer_t.cpp", crs, {H / 32}, dm_config(0, 0));
            writer.emplace_common_runtime_args({outputs[0].buffer(), total, per, gx});
        } else {
            writer = kernel_desc("reduce_writer.cpp", crs, {RB, TILES, S, uint32_t(args.split)}, dm_config(0, 0));
            KernelDescriptor::RTArgList w;
            w.push_back(outputs[0].buffer());
            if (args.split) {
                w.push_back(outputs[1].buffer());
            } else {
                w.push_back(0u);
            }
            w.push_back(t.chip_info.buffer());
            w.push_back(total);
            w.push_back(per);
            w.push_back(gx);
            writer.emplace_common_runtime_args(w);
        }
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(compute));
        desc.kernels.push_back(std::move(writer));
        return desc;
    }

    auto reader = kernel_desc("reduce2_reader.cpp", crs, {K, RB, TILES, 64, args.phase}, dm_config(1, 1));
    KernelDescriptor::RTArgList r;
    r.push_back(t.y.buffer());
    r.push_back(t.y_slot.buffer());
    r.push_back(t.weights.buffer());
    r.push_back(total);
    r.push_back(per);
    r.push_back(t.chip_info.buffer());
    if (t.peer.has_value()) {
        r.push_back(t.peer->buffer());
    } else {
        r.push_back(0u);
    }
    r.push_back(gx);
    reader.emplace_common_runtime_args(r);
    // phase 2 tiled: the same compute / writer as phase 0 tiled (the reader's per-token header + pairs protocol is
    // shared; the peer's partial is one more pair), the [S, H] own partial as tiles (the TP reduce-scatter input)
    auto compute =
        kernel_desc(tiled ? "reduce_compute_t.cpp" : "reduce_compute.cpp", crs, {TILES}, fp32_compute_config());
    compute.emplace_common_runtime_args(rng);
    KernelDescriptor writer;
    if (tiled) {
        writer = kernel_desc("reduce_writer_t.cpp", crs, {H / 32}, dm_config(0, 0));
        writer.emplace_common_runtime_args({outputs[0].buffer(), total, per, gx});
    } else {
        writer = kernel_desc("reduce_writer.cpp", crs, {RB, TILES, 1, 0}, dm_config(0, 0));
        writer.emplace_common_runtime_args({outputs[0].buffer(), 0u, 0u, total, per, gx});
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    return desc;
}

}  // namespace ttnn::prim
