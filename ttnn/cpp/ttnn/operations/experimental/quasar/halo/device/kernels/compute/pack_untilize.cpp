// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/compute/pack_untilize.h"
#include "experimental/kernel_args.h"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;
constexpr uint32_t NUM_RISCV_DATA_MOVEMENT_CORES = 2;

void kernel_main() {
    // Metal 2.0 named bindings:
    //   dfb::src           - borrowed input shard (compute consumes, untilizes)
    //   dfb::untilize_out0 - untilized output for reader0 (even blocks)
    //   dfb::untilize_out1 - untilized output for reader1 (odd blocks)
    constexpr uint32_t src_cb_id = dfb::src;
    constexpr uint32_t out_cb_id0 = dfb::untilize_out0;
    constexpr uint32_t out_cb_id1 = dfb::untilize_out1;
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);  // number of tiles along width of shard
    constexpr uint32_t block_size = get_arg(args::block_size);  // number of tiles along height that make up a block

    const uint32_t total_blocks = get_arg(args::total_blocks);

    DataflowBuffer src_cb(src_cb_id);
    DataflowBuffer out_cb0(out_cb_id0);
    DataflowBuffer out_cb1(out_cb_id1);

    compute_kernel_hw_startup(src_cb_id, out_cb_id0);

#ifdef ARCH_QUASAR
    // Quasar: the pack untilize LLK bakes the destination DFB's buffer descriptor id into the pack MOP at
    // pack_untilize_init, so the packer must be re-targeted whenever the output alternates between out0 (reader0's
    // even blocks) and out1 (reader1's odd blocks).
    //
    // That re-target, and the per-row UNTILIZE_DST_ADDR_OFFSET write inside llk_pack_untilize, are RISC MMIO writes
    // (MOP bank-0 config, buffer descriptor table, THCON packer cfg). They are NOT ordered against the PACRs of the
    // previous row: the pack TRISC issues those PACRs, their SEMWAIT on math and the TT_PUSH_TILES as Tensix
    // instructions and runs straight on, and mop_sync() only waits for a MOP that has already reached the expander,
    // not for one still queued behind the SEMWAIT. The previous row's PACRs could then execute with the next row's
    // MOP / BD / L1 offset, so part of a 32-row block (e.g. the second tile = channels 32..63) went to the wrong
    // DFB or offset and the reader sent whatever was left in that L1 slot from an earlier program (stale data that
    // changes from run to run, mostly in the first block, when the pack TRISC is furthest ahead).
    //
    // Fix: initialize unpack + math once (the source DFB, its format and shape never change), and on the pack
    // thread drain the Tensix pipeline (tensix_sync) before every pack-side config write, i.e. before each re-target
    // and each pack_untilize_block.
    using untilize_dispatch = compute_kernel_lib::UntilizeDispatchConfig<tiles_per_row, src_cb_id, out_cb_id0>;
    constexpr uint32_t sub_block_w = untilize_dispatch::sub_block_width;
    constexpr uint32_t num_sub_blocks = untilize_dispatch::num_sub_blocks;
    static_assert(sub_block_w * num_sub_blocks == tiles_per_row, "untilize sub-blocks must cover the row");

    pack_untilize_init<sub_block_w, tiles_per_row>(src_cb_id, out_cb_id0);
    uint32_t packer_out_cb_id = out_cb_id0;

    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        const bool even_block = (block_idx % 2) == 0;
        const uint32_t out_cb_id = even_block ? out_cb_id0 : out_cb_id1;
        auto& out_cb = even_block ? out_cb0 : out_cb1;
        for (uint32_t r = 0; r < block_size; ++r) {
            out_cb.reserve_back(tiles_per_row);
            for (uint32_t b = 0; b < num_sub_blocks; ++b) {
                src_cb.wait_front(sub_block_w);
                // Wait for the previous row's PACRs / push to complete before touching packer config.
                PACK((ckernel::tensix_sync()));
                if (out_cb_id != packer_out_cb_id) {
                    pack_untilize_dest_init<sub_block_w, tiles_per_row>(out_cb_id);
                    packer_out_cb_id = out_cb_id;
                }
                pack_untilize_block<sub_block_w, tiles_per_row>(src_cb_id, 1, out_cb_id, b);
                src_cb.pop_front(sub_block_w);
            }
            out_cb.push_back(tiles_per_row);
        }
    }
    pack_untilize_uninit(packer_out_cb_id);
#else
    // QSR FIX: the Quasar pack_untilize LLK bakes the destination CB's base L1 address at
    // pack_untilize_init time (it ignores the runtime output CB argument thereafter). The split
    // reader alternates the untilized output between two CBs (out0 for reader0's even blocks, out1
    // for reader1's odd blocks). With a single up-front init on out0 (Neither mode in the loop), the
    // odd blocks packed for out1 land in out0's baked address instead — so out1 stays empty and
    // reader1 scatter-writes zeros (half the halo output is lost on multi-core). Re-init the packer
    // per block for its actual destination CB so the baked address is correct each iteration.
    // (On WH/BH the packer honours the runtime output CB, so the shared code path uses Neither; this
    // per-block re-init is the Quasar-correct equivalent.)
    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        if (block_idx % 2 == 0) {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id0,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
                src_cb, out_cb0, block_size);
        } else {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id1,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
                src_cb, out_cb1, block_size);
        }
    }
#endif
}
