// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// OPTION B — UNPACK-TILIZE PROBE for the Quasar conv2d.
//
// The plain tilize path (tilize_init/tilize_block, used by conv_tilize_only_metal2.cpp and the standalone
// tilize op) 0x19s mid-stream on Quasar (a DEST-bank fault in tilize_block — intrinsic LLK bug). This probe
// runs the SAME conv gathered activation (dfb::act) through the DIFFERENT tilize path used by the Quasar
// MAXPOOL compute — `unpack_tilizeA_B_block` + `reduce_tile_math` — which the pool uses successfully with no
// 0x19. Goal: (a) confirm the unpack-tilize path processes the conv activation WITHOUT the 0x19, and
// (b) localize the bug to tilize_block vs the unpack-tilize path.
//
// This is a PROBE: the OUTPUT is throwaway (a row-reduce collapse, packed into a plain ACT_TILIZED DFB). The
// ONLY signal is whether it completes without a 0x19 and how many chunks it processes (UTPROBE DPRINT). The
// reduce scalar (srcB) is credit-only (unfilled) — its VALUE cannot cause a MOP-issue fault (0x19), so a
// garbage scalar is fine for a completion probe; it keeps the srcB CB valid without a cross-thread fill.
//
// API sequence mirrors ttnn/.../pool_generic/device/kernels/compute/compute_pool_2d.cpp exactly:
//   tilizeA_B_reduce_init<neginf,zero>(inA, scalar, block, out)            (once, does the hw_configure)
//   pack_untilize_dest_init<CHUNK>(out)                                     (once)
//   per chunk: tilizeA_B_reduce_init_short<neginf,zero>(inA, scalar, block, out)  (no hw_configure)
//              tile_regs_acquire()
//              unpack_tilizeA_B_block<neginf,reload_srcB=true,zero_srcA=false,zero_reduce=false>(inA, scalar, block, 0)
//              reduce_tile_math<REDUCE_OP, REDUCE_DIM>(t, num_faces)  for t in 0..block
//              tile_regs_commit(); tile_regs_wait()
//              pack_untilize_dest<CHUNK>(out, 1, 0); push
//              tile_regs_release()
//
// Per LLK: unpack_tilizeA_B is REDUCE-ONLY on Quasar — reload_srcB must be true, zero_srcA false, srcA is
// tilized + srcB is a SCALAR; you cannot reverse srcA/srcB or feed a matmul. Selected by the sharded factory
// under TT_METAL_QSR_CONV_UNPACK_TILIZE for the resnet stem / 1x1 height-sharded single-K-block shape.

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/reduce.h"
#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "api/debug/dprint.h"

// MAX reduce: neginf srcA padding, no zero srcA (mirror the pool's maxpool config).
static constexpr bool neginf_srca = true;
static constexpr bool zero_srca = false;
// num faces in a full 32x32 input tile (pool uses 4; on Quasar reduce_tile_math reads face geometry from CB
// metadata but keep it coherent).
static constexpr uint32_t num_faces_in_input_tile = 4;

void kernel_main() {
    constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    constexpr uint32_t reader_num_h_subblocks = get_arg(args::reader_num_h_subblocks);
    constexpr uint32_t in0_num_blocks_h = get_arg(args::in0_num_blocks_h);
    constexpr bool height_sharded = get_arg(args::height_sharded);

    constexpr uint32_t in_cb_id = dfb::act;               // gathered im2col activation (row-major sticks)
    constexpr uint32_t in_scalar_cb_id = dfb::in_scalar;  // reduce scalar (srcB), credit-only for the probe
    constexpr uint32_t out_cb_id = dfb::act_tilized;      // throwaway reduced output (plain DFB)

    // Process the whole per-core gathered activation as a flat tile stream, chunked to fit DEST. The conv
    // reader produced in0_num_blocks_h row-blocks of (reader_num_h_subblocks * in0_block_w) tiles each; total
    // per core = M*K tiles = num_blocks_row * in0_block_w. CHUNK <= 8 keeps a half-sync DEST (8 tiles) valid;
    // in0_block_w (K) is a multiple of CHUNK for the stem (K=16), so chunks tile the stream evenly.
    constexpr uint32_t CHUNK = in0_block_w < 8 ? in0_block_w : 8;
    static_assert(CHUNK > 0, "CHUNK must be > 0");
    static_assert(in0_block_w % CHUNK == 0, "in0_block_w must be a multiple of CHUNK for even chunking");
    constexpr uint32_t num_blocks_row = in0_num_blocks_h * reader_num_h_subblocks;  // = M (tile-rows)
    constexpr uint32_t total_tiles = num_blocks_row * in0_block_w;                  // = M*K
    constexpr uint32_t num_chunks = total_tiles / CHUNK;

    if constexpr (!height_sharded) {
        return;  // probe only wired for the height-sharded conv path
    }

    // Scalar srcB: provide a credit so unpack_tilizeA_B's srcB read is gated correctly. Compute self-loop
    // (PRODUCER + CONSUMER binding): reserve+push once, then wait once, never pop (one-scalar-per-core style).
    // The L1 content is intentionally unfilled (throwaway reduce). DataflowBuffer methods thread-gate
    // internally (reserve/push -> PACK, wait/pop -> UNPACK).
    DataflowBuffer in_scalar_cb(in_scalar_cb_id);
    DataflowBuffer in_cb(in_cb_id);
    DataflowBuffer out_cb(out_cb_id);

    in_scalar_cb.reserve_back(1);
    in_scalar_cb.push_back(1);
    in_scalar_cb.wait_front(1);

    // One-time init (does the llk_*_hw_configure): mirror the pool's kernel-start init.
    tilizeA_B_reduce_init<neginf_srca, zero_srca>(in_cb_id, in_scalar_cb_id, CHUNK, out_cb_id);
    pack_untilize_dest_init<CHUNK>(out_cb_id);

    UNPACK(DPRINT(
        "UTPROBE start K={} CHUNK={} num_chunks={} total={}\n",
        (uint32_t)in0_block_w,
        (uint32_t)CHUNK,
        (uint32_t)num_chunks,
        (uint32_t)total_tiles));

    for (uint32_t iter = 0; iter < num_chunks; ++iter) {
        // Re-init unpack+math for this chunk WITHOUT re-running hw_configure (the *_short variant). Re-running
        // hw_configure per chunk corrupts unpacker state on Quasar (pool comment) — this keeps unpack+math in
        // lockstep, exactly as the pool does per c-block.
        tilizeA_B_reduce_init_short<neginf_srca, zero_srca>(in_cb_id, in_scalar_cb_id, CHUNK, out_cb_id);

        tile_regs_acquire();
        in_cb.wait_front(CHUNK);
        unpack_tilizeA_B_block<neginf_srca, true /*reload_srcB*/, false /*zero_srcA*/, zero_srca>(
            in_cb_id, in_scalar_cb_id, CHUNK, 0 /*srcB tile idx*/);
        for (uint32_t t = 0; t < CHUNK; ++t) {
            reduce_tile_math<REDUCE_OP, REDUCE_DIM>(t, num_faces_in_input_tile);
        }
        in_cb.pop_front(CHUNK);
        tile_regs_commit();
        tile_regs_wait();

        out_cb.reserve_back(CHUNK);
        pack_untilize_dest<CHUNK>(out_cb_id, 1, 0);
        out_cb.push_back(CHUNK);
        tile_regs_release();

        // Per-chunk progress: the LAST UTPROBE line before a fault shows exactly how far the unpack-tilize
        // path got (compare to conv_tilize_only, which 0x19s after ~5 tilize_block blocks).
        UNPACK(DPRINT("UTPROBE iter={}/{} done\n", (uint32_t)(iter + 1), (uint32_t)num_chunks));
    }
    UNPACK(DPRINT("UTPROBE COMPLETE all {} chunks (no 0x19)\n", (uint32_t)num_chunks));
}  // void kernel_main()
