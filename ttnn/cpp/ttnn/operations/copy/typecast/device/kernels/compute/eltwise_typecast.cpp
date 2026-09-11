// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "api/debug/dprint.h"         // [TC-DBG #51270] remove after typecast multi-tile isolation
#include "api/debug/dprint_tensix.h"  // [TC-DBG #51270] DEST-register dump (dprint_tensix_dest_reg_row_float32)

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_dim = get_arg(args::per_core_block_dim);

    // dfb::in  — the typecast source pages, filled by this factory's reader
    // dfb::out — the typecast result pages, drained by the writer (or, on the sharded path,
    //            resident in the borrowed output buffer with no writer to drain it)
    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_init(dfb::in);
    // [TC-DBG #51270] per_core_block_cnt distinguishes the two grids: it is 2 on the single-compute-node
    // 1x3 grid (both tiles on one core -> the failing path) and 1 on the 2x3 grid (one tile per core).
    // Unguarded DPRINT (matches the eltwise compute-kernel idiom); may print once per active TRISC.
    // sync: DstSync value the kernel compiled with (SyncHalf=0, SyncFull=1). Hypothesis: on the emulator
    // this metal2 (QuasarComputeConfig) kernel falls back to SyncFull=1, which quirk #3 (TEN-4636) says
    // zeros the upper DEST half -> the 2nd tile. accum: DST_ACCUM_MODE (fp32-dest = 1). tile 0 being
    // bit-exact fp32 means accum must already be 1; if sync also prints 1 that confirms the Full trap.
    DPRINT("TC_CMP cnt={} sync={} accum={}\n", per_core_block_cnt, (uint32_t)DST_SYNC_MODE, (uint32_t)DST_ACCUM_MODE);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // [TC-DBG #51270] block marker: confirms the loop runs twice on 1x3 and correlates the per-tile
        // TYPECAST_LLK_INIT below with the tile the writer reports as wrong.
        DPRINT("TC_CMP blk={}\n", block_index);
        dfb_out.reserve_back(per_core_block_dim);
#ifdef ARCH_QUASAR
        // [#51270] Re-point the packer's Buffer Descriptor at the current out_cb ring slot for EACH
        // output block. compute_kernel_hw_startup runs llk_pack_init only once, and on Quasar the pack
        // BD (which holds the L1 slot address) does not auto-advance across DFB ring slots. Without this
        // the 2nd block packs to the already-consumed 1st slot, so out slot 1 is never written and reads
        // back as zeros (confirmed by DPRINT: tile 1 fp32bits = 0). Same per-use pack_init the Quasar
        // pool kernel uses; WH/BH need no repoint (pack_tile tracks the CB slot there).
        pack_init(dfb::out);
#endif
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            dfb_in.wait_front(1);

            copy_tile(dfb::in, 0, 0);

            // [TC-DBG #51270] STEP 1 — DEST row 0 right after copy_tile (before the SFPU). Shows whether
            // the unpacker actually loaded this block's input into the active DEST bank. If block 1 is
            // zero here, the copy/unpack (not the SFPU) is the failure.
            MATH((DPRINT("TC_DST_COPY blk={}\n", block_index)));
            MATH((dprint_tensix_dest_reg_row_float32(0)));

            TYPECAST_LLK_INIT();
            TYPECAST_LLK(0);

            // [TC-DBG #51270] STEP 2 — DEST row 0 right after the typecast SFPU. If block 1 was non-zero
            // after copy (STEP 1) but zero here, the SFPU typecast is zeroing the second tile.
            MATH((DPRINT("TC_DST_TC blk={}\n", block_index)));
            MATH((dprint_tensix_dest_reg_row_float32(0)));

            tile_regs_commit();

            tile_regs_wait();

            pack_tile(0, dfb::out);

            dfb_in.pop_front(1);

            tile_regs_release();
        }
        dfb_out.push_back(per_core_block_dim);
    }
}
