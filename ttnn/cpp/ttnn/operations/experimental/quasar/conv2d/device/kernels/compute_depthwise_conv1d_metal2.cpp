// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of compute_depthwise_conv1d.cpp (1D depthwise conv2d compute kernel).
// Algorithm body identical to the legacy kernel; only the host-binding surface is migrated:
//   - CB-index CTAs -> dfb:: tokens (act / weights / act_tilized / out)
//   - remaining positional CTAs -> get_arg(args::name)
//   - experimental::CB -> DataflowBuffer (kernel_main + helper signatures)

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// Compute one block (one kernel-tap slice) of a 1D depthwise conv.
//
// Per output tile:
//   dst[0]  = in0 * in1                                                 (FPU mul)
//   if idx > 0: dst[0] += prior partial loaded from out_cb              (FPU add via DST reuse)
//   pack dst[0] -> out_cb
//
// `idx` is the kernel-tap block index (0 .. filter_h*filter_w-1). The very first call (idx == 0)
// initializes out_cb with the tap-0 product; subsequent calls accumulate via the DST_TO_SRCB
// dest-reuse pattern, which keeps the running partial in DST and only pulls the prior partial
// from L1. This gives a single pack per output tile (to out_cb) and avoids the pack-format flips
// that corrupt block-float (BFLOAT8_B/BFLOAT4_B) outputs in the round-tripped variant — while
// still using FPU (not SFPU) for the add.
//
// srcB (cfg92) tile descriptor: must match in1 for the mul, and is repopulated from DST for the
// dest-reuse add. We force srcB back to in1's format every iteration so block-float weights are
// decoded correctly.
inline void mul_and_accumulate_block(
    DataflowBuffer in0_cb, DataflowBuffer in1_cb, DataflowBuffer out_cb, uint32_t block_num_tiles, uint32_t idx) {
    const uint32_t in0_cb_id = in0_cb.get_id();
    const uint32_t in1_cb_id = in1_cb.get_id();
    const uint32_t out_cb_id = out_cb.get_id();

    for (uint32_t i = 0; i < block_num_tiles; i++) {
        in1_cb.wait_front(1);
        in0_cb.wait_front(1);

        tile_regs_acquire();
        // mul: srcA = in0 (bf16), srcB = in1 (bf8/bf16) -> dst[0]
        reconfig_data_format_srcb(in1_cb_id);
        mul_init(in0_cb_id, in1_cb_id);
        mul_tiles(in0_cb_id, in1_cb_id, 0, 0, 0);

        if (idx != 0) {
            // dest-reuse add: dst[0] += out_cb. srcA gets out_cb (cfg52 must match out_cb fmt);
            // srcB is filled from dst[0] by the dest-reuse path.
            reconfig_data_format_srca(out_cb_id);
            add_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(out_cb_id, out_cb_id);
            out_cb.wait_front(1);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                out_cb_id, 0, 0);
            out_cb.pop_front(1);

            // Restore srcA to in0's format for the next iteration's mul unpack.
            reconfig_data_format_srca(in0_cb_id);
        }
        tile_regs_commit();

        out_cb.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, out_cb_id);
        out_cb.push_back(1);
        tile_regs_release();

        in0_cb.pop_front(1);
        in1_cb.pop_front(1);
    }
}

template <uint32_t in0_block_w, uint32_t kernel_width, uint32_t block_num_tiles>
inline void mul_and_accumulate_coalesced_block(DataflowBuffer in0_cb, DataflowBuffer in1_cb, DataflowBuffer out_cb) {
    static_assert(kernel_width > 1);
    static_assert(in0_block_w % kernel_width == 0);
    static_assert(block_num_tiles % in0_block_w == 0);

    constexpr uint32_t in_channels_ntiles = in0_block_w / kernel_width;
    constexpr uint32_t act_block_h_ntiles = block_num_tiles / in0_block_w;

    const uint32_t in0_cb_id = in0_cb.get_id();
    const uint32_t in1_cb_id = in1_cb.get_id();
    const uint32_t out_cb_id = out_cb.get_id();

    in0_cb.wait_front(block_num_tiles);
    in1_cb.wait_front(block_num_tiles);

    for (uint32_t h = 0; h < act_block_h_ntiles; ++h) {
        for (uint32_t c = 0; c < in_channels_ntiles; ++c) {
            tile_regs_acquire();
            reconfig_data_format_srca(in0_cb_id);
            reconfig_data_format_srcb(in1_cb_id);

            for (uint32_t tap = 0; tap < kernel_width; ++tap) {
                const uint32_t act_tile_idx = h * in0_block_w + tap * in_channels_ntiles + c;
                const uint32_t weight_tile_idx =
                    tap * act_block_h_ntiles * in_channels_ntiles + h * in_channels_ntiles + c;
                mul_init(in0_cb_id, in1_cb_id, tap != 0 ? 1U : 0U, __builtin_LINE());
                mul_tiles(in0_cb_id, in1_cb_id, act_tile_idx, weight_tile_idx, 0);
            }
            tile_regs_commit();

            out_cb.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, out_cb_id);
            out_cb.push_back(1);
            tile_regs_release();
        }
    }

    in0_cb.pop_front(block_num_tiles);
    in1_cb.pop_front(block_num_tiles);
}

void kernel_main() {
    constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    constexpr uint32_t in0_num_subblocks = get_arg(args::in0_num_subblocks);
    constexpr uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr uint32_t in0_num_blocks_h = get_arg(args::in0_num_blocks_h);
    constexpr uint32_t in0_num_blocks_w = get_arg(args::in0_num_blocks_w);
    constexpr uint32_t in0_cb_id = dfb::act;
    constexpr uint32_t in1_cb_id = dfb::weights;
    constexpr uint32_t tilized_in0_cb_id = dfb::act_tilized;
    constexpr uint32_t out_cb_id = dfb::out;
    constexpr uint32_t kernel_width = get_arg(args::kernel_width);
    constexpr bool coalesce_kw_reads = get_arg(args::coalesce_kw_reads) == 1;

    DataflowBuffer cb_tilized_in0(tilized_in0_cb_id);
    DataflowBuffer cb_in1(in1_cb_id);
    DataflowBuffer cb_out(out_cb_id);

    // compute_kernel_hw_startup configures pack for out_cb, math for in0/in1, and unpack for in0/in1.
    // The pack target never changes (we only ever pack to out_cb), so no further pack reconfig is
    // needed for the lifetime of the kernel.
    compute_kernel_hw_startup(in0_cb_id, in1_cb_id, out_cb_id);

    for (uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
        for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
            compute_kernel_lib::tilize<in0_block_w, in0_cb_id, tilized_in0_cb_id>(in0_num_subblocks);
            reconfig_data_format_srca(tilized_in0_cb_id);
            if constexpr (coalesce_kw_reads) {
                mul_and_accumulate_coalesced_block<in0_block_w, kernel_width, in0_block_num_tiles>(
                    cb_tilized_in0, cb_in1, cb_out);
            } else {
                const uint32_t idx = in0_block_h_i * in0_num_blocks_w + in0_block_w_i;
                mul_and_accumulate_block(cb_tilized_in0, cb_in1, cb_out, in0_block_num_tiles, idx);
            }
        }
    }
}
