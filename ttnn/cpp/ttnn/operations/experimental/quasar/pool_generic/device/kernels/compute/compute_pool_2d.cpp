// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/reduce.h"
#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/add_int_sfpu.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "api/debug/ring_buffer.h"  // DEBUG pool compute-stall: ring-buffer markers (remove after)

#define DEBUG_PRINT 0  // [DEBUG scratch-pack experiment] (remove after)

// [DEBUG] Pack-target selector for the ROW_MAJOR path (remove after):
//   0 = production path: narrow pack_untilize straight into the real output CB (out_cb), then DPRINT it.
//   1 = experiment:      full-tile (32x32) pack of the reduced DEST into the scratch CB, then DPRINT it
//                        (out_cb still gets a balancing push with garbage).
// The inits (tilizeA_B_reduce_init + pack_untilize_dest_init) and the pack call all follow this switch,
// so flipping this one value moves the whole pipeline between the two CBs consistently.
// NOTE: PACK_TO_SCRATCH==1 requires the reader-side scratch consume (reader_pool_2d.cpp) to be enabled
// too — compute PRODUCES the scratch CB and the DM reader CONSUMES it; they must be on/off together.
#define PACK_TO_SCRATCH 1

#if DEBUG_PRINT == 1
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
// NOTE: dprint_tensix.h pulls in ckernel_debug.h which does not exist on Quasar — omit it (we only
// need DPRINT + print_bf16_pages here). (remove after)
#endif

#define ALWI inline __attribute__((always_inline))

#define FACE_HEIGHT 16
#define FACE_WIDTH 16
#define TILE_HEIGHT 32
#define TILE_WIDTH 32

void kernel_main() {
#if DEBUG_PRINT == 1
    PACK(DPRINT("POOL2D_ENTER (compute kernel_main reached)\n"));
    UNPACK(DPRINT("POOL2D_ENTER_UNPACK\n"));
    MATH(DPRINT("POOL2D_ENTER_MATH\n"));
#endif
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet. When ntiles_hw > 1 the large
    // kernel is called
    constexpr uint32_t in_ntiles_c = get_arg(args::in_ntiles_c);
    constexpr uint32_t window_size_hw = get_arg(args::window_size_hw);
    constexpr uint32_t scratch_npages = get_arg(args::scratch_npages);  // [DEBUG scratch->out] whole-CB count

    constexpr uint32_t split_reader = get_arg(args::split_reader);

    constexpr uint32_t max_out_sticks_per_core = get_arg(args::max_out_sticks_per_core);
    constexpr uint32_t in_c = get_arg(args::in_c);
    constexpr uint32_t in_nblocks_c = get_arg(args::in_nblocks_c);
    constexpr uint32_t max_sticks_for_reduction = get_arg(args::max_sticks_for_reduction);

    // CB ids are Metal 2.0 DFB tokens. Keep the legacy variable names so the rest of the
    // kernel (DataflowBuffer construction and LLK calls taking a uint32_t CB id) is unchanged;
    // dfb::<name> converts implicitly to uint32_t.
    constexpr auto in_cb_id_0 = dfb::in_cb_0;
#ifdef SPLIT_READER
    constexpr auto in_cb_id_1 = dfb::in_cb_1;  // for split reader
#endif
    constexpr auto in_scalar_cb_id_0 = dfb::in_scalar_cb_0;
#ifdef SPLIT_READER
    constexpr auto in_scalar_cb_id_1 = dfb::in_scalar_cb_1;
#endif
    constexpr auto out_cb_id = dfb::out_cb;
    constexpr auto scratch_cb_id_0 = dfb::scratch_cb_0;  // [DEBUG scratch-pack] per-reader scratch targets
#ifdef SPLIT_READER
    constexpr auto scratch_cb_id_1 = dfb::scratch_cb_1;
#endif
    constexpr bool one_scalar_per_core = get_arg(args::one_scalar_per_core);
    constexpr bool is_output_tiled = get_arg(args::is_output_tiled);  // 1 = TILED, 0 = ROW_MAJOR
    constexpr bool is_output_block_format = (bool)get_arg(args::is_output_block_format);
#ifdef OUTPUT_TILED
    constexpr auto pre_tilize_cb_id = dfb::pre_tilize_cb;
    // fast_tilize_cb_id is a consumer-view alias of pre_tilize_cb_id (same L1 region,
    // full-tile face_geometry = {face_r_dim=16, num_faces=4}). Used as the input operand
    // to fast_tilize so the unpacker/math read the correct face count from CB metadata.
    constexpr auto fast_tilize_cb_id = dfb::fast_tilize_cb;
#endif

    constexpr bool use_split_reader = split_reader;

    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    // QSR: match num_faces_in_input_tile_for_cb in the pool factory. The reduce-col strided tilize
    // requires a full 32x32 (4-face) SrcA tile, so always reduce 4 faces; padding rows [window,32) hold
    // the pool identity so the extra reduced rows are a no-op. (On Quasar reduce_tile_math ignores this
    // num_faces arg and uses the CB face-geometry metadata, but keep it coherent.)
    constexpr uint32_t num_faces_in_input_tile = 4;
    // "Single partial tile per core that fits in one face": when there is only one output tile
    // per core (in_c < TILE_WIDTH) and it fits in a single face (in_c <= FACE_WIDTH), pack just
    // one face for that tile. The host correspondingly aligns output_shard_width to FACE_WIDTH,
    // so the per-shard layout has no internal padding and downstream consumers (e.g.
    // sharded_to_interleaved) read contiguous data.
    constexpr bool single_partial_fits_in_face = last_tile_is_partial && in_c <= FACE_WIDTH;
    constexpr uint32_t num_faces_in_output_tile = single_partial_fits_in_face ? 1 : 2;
    // When the last tile has exactly FACE_WIDTH valid channels (channels % 32 == 16) OR the only
    // tile is partial-fits-in-one-face, pack 1 face for the last tile.
    constexpr uint32_t num_faces_in_last_output_tile =
        last_tile_is_partial && (in_c % TILE_WIDTH == FACE_WIDTH || single_partial_fits_in_face) ? 1 : 2;

    constexpr bool is_avg_pool = REDUCE_OP == PoolType::AVG;
    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time. Callers (e.g. grid_sample under fp32_dest_acc_en) can
    // also force the 4-tile limit via ct_arg[16] so each chunk fits in half-sync DEST (= 4 fp32 tiles)
    // without forcing dst_full_sync_en.
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr bool force_max_tiles_per_reduction_4 = get_arg(args::force_max_tiles_per_reduction_4);
    constexpr uint32_t MAX_TILES_PER_REDUCTION =
        (force_max_tiles_per_reduction_4 || (is_avg_pool && is_large_kernel)) ? 4 : 8;
    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    static_assert(REDUCE_OP == PoolType::MAX || REDUCE_OP == PoolType::AVG, "Only supports REDUCE_OP = MAX or AVG");
    constexpr bool neginf_srca_maxpool = (REDUCE_OP == PoolType::MAX) ? true : false;
    constexpr bool zero_srca_avgpool = (REDUCE_OP == PoolType::AVG) ? true : false;

    // tilize reconfiguration can be beneficial when we have a wide tensor with a non MAX_TILES_PER_REDUCTION number of
    // C tiles, but we only use it when the window size fits within a face such that the tilize can be done only on the
    // rows populated with data, otherwise we need to call clear_out_tiles between reconfigs to avoid untilizing junk
    // data which is much slower than just untilizing the entire MAX_TILES_PER_REDUCTION
    constexpr bool tilize_reconfig = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 &&
                                     window_size_hw <= FACE_HEIGHT && !last_tile_is_partial;

    // tilize_untilize_cb references pre_tilize_cb_id at parse time, so gate the name selection
    // at the preprocessor (an `is_output_tiled ? ...` ternary would still name-look-up
    // pre_tilize_cb_id on the non-tiled build, where that DFB token is not emitted).
#ifdef OUTPUT_TILED
    constexpr uint32_t tilize_untilize_cb = pre_tilize_cb_id;
#else
    constexpr uint32_t tilize_untilize_cb = out_cb_id;
#endif

    DataflowBuffer in_scalar_cb_0(in_scalar_cb_id_0);
    DataflowBuffer in_cb_0(in_cb_id_0);
#ifdef SPLIT_READER
    DataflowBuffer in_scalar_cb_1(in_scalar_cb_id_1);
    DataflowBuffer in_cb_1(in_cb_id_1);
#endif
    DataflowBuffer out_cb(out_cb_id);
    DataflowBuffer scratch_cb_0(scratch_cb_id_0);  // [DEBUG scratch-pack]
#ifdef SPLIT_READER
    DataflowBuffer scratch_cb_1(scratch_cb_id_1);
#endif
#ifdef OUTPUT_TILED
    DataflowBuffer pre_tilize_cb(pre_tilize_cb_id);
    DataflowBuffer fast_tilize_cb(fast_tilize_cb_id);
#endif

    // [DEBUG] Pack-target CB (follows PACK_TO_SCRATCH). The reduce init and the pack_untilize init MUST
    // target the same CB or the pipeline desyncs. tilize_untilize_cb == out_cb_id for the RM build.
#if PACK_TO_SCRATCH == 1
    // Both scratch CBs share the same full-tile geometry, so init once with scratch_cb_0.
    constexpr uint32_t pack_target_cb_id = scratch_cb_id_0;
#else
    constexpr uint32_t pack_target_cb_id = tilize_untilize_cb;
#endif
    tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
        in_cb_id_0, in_scalar_cb_id_0, max_tiles_per_iter, pack_target_cb_id);

    pack_untilize_dest_init<max_tiles_per_iter>(pack_target_cb_id);

    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;

    // wait for initialization to complete
    if constexpr (one_scalar_per_core) {
        in_scalar_cb_0.wait_front(1);
    }

    // if max out sticks is non-zero then this will be used as the number of out sticks for every core
    // otherwise the runtime args are referenced for core-specific number of out sticks, for Pool2D
    // runtime args are used while for grid sample the max out sticks is set
    uint32_t num_out_sticks_this_core =
        max_out_sticks_per_core ? max_out_sticks_per_core : get_arg(args::out_nhw_this_core);
    uint32_t last_tile_height =
        num_out_sticks_this_core % TILE_HEIGHT == 0 ? TILE_HEIGHT : num_out_sticks_this_core % TILE_HEIGHT;

    uint32_t tilize_stick_counter = 0;
    uint32_t tilize_stick_total = 0;
#if DEBUG_PRINT == 1
    PACK(DPRINT(
        "POOLCOMPUTE tiled={} nsticks={} in_nblocks_c={}\n",
        (uint32_t)is_output_tiled,
        num_out_sticks_this_core,
        (uint32_t)in_nblocks_c));
#endif
    for (uint32_t n = 0; n < num_out_sticks_this_core; ++n) {
        const bool reader0 = !(use_split_reader && (n & 0x1));
        const bool use_reader1_scalar = !reader0 && !one_scalar_per_core;
        // The reader1 (split) DFB tokens only exist under SPLIT_READER; gate the selection at
        // the preprocessor so the non-split build never names dfb::in_cb_1 / dfb::in_scalar_cb_1.
#ifdef SPLIT_READER
        const uint32_t curr_scalar_cb_id = use_reader1_scalar ? in_scalar_cb_id_1 : in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = !reader0 ? in_cb_id_1 : in_cb_id_0;
        DataflowBuffer curr_scalar_cb = use_reader1_scalar ? in_scalar_cb_1 : in_scalar_cb_0;
        DataflowBuffer curr_in_cb = reader0 ? in_cb_0 : in_cb_1;
#else
        const uint32_t curr_scalar_cb_id = in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = in_cb_id_0;
        DataflowBuffer curr_scalar_cb = in_scalar_cb_0;
        DataflowBuffer curr_in_cb = in_cb_0;
#endif
        if constexpr (!one_scalar_per_core) {
            curr_scalar_cb.wait_front(1);
        }
        if (is_output_tiled && !tilize_stick_counter) {
            out_cb.reserve_back(in_ntiles_c);
        }
        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            const bool last_c_block = c_i == in_nblocks_c - 1;
            const bool first_c_block = c_i == 0;
            const uint32_t tiles_to_reduce =
                tilize_reconfig ? (last_c_block ? partial_iter_output_tiles : max_tiles_per_iter) : max_tiles_per_iter;
            const uint32_t number_of_tiles = last_c_block ? partial_iter_output_tiles : max_tiles_per_iter;
            const uint32_t output_faces =
                (last_tile_is_partial && last_c_block &&
                 (in_c % TILE_WIDTH == FACE_WIDTH || single_partial_fits_in_face))
                    ? (number_of_tiles - 1) * num_faces_in_output_tile + num_faces_in_last_output_tile
                    : number_of_tiles * num_faces_in_output_tile;
            // [DEBUG pool compute stall] Which call blocks the WFD/UPTW deadlock? Newest ring marker per
            // thread: 0xC0FFEE10 -> out_cb.reserve_back (output CB full); 0xC0FFEE11 -> tile_regs_acquire
            // (dest register busy); 0xC0FFEE12 -> curr_in_cb.wait_front (input starved — reader can't fill);
            // 0xC0FFEE13 -> tile_regs_wait (math never committed the reduce). n/c_i/chunk give the position.
#if PACK_TO_SCRATCH == 0
            if constexpr (!is_output_tiled) {
                PACK(WATCHER_RING_BUFFER_PUSH(0xC0FFEE10u));
                PACK(WATCHER_RING_BUFFER_PUSH((uint32_t)n));
                PACK(WATCHER_RING_BUFFER_PUSH((uint32_t)c_i));
                out_cb.reserve_back(output_faces);
            }
#endif
            if constexpr (tilize_reconfig) {
                if (first_c_block || last_c_block) {
                    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce)));
                }
            }
            // QSR split-reader fix: the strided reduce-col unpack (_llk_unpack_reduce_col_tilizeA_strided_)
            // binds its input buffer descriptor at init time (tilizeA_B_reduce_init used in_cb_0). On Quasar
            // that descriptor is NOT re-derived from the CB id passed to unpack_tilizeA_B_block below, so the
            // odd (reader1) sticks otherwise keep reading in_cb_0 and march past its filled page into zeros
            // (proven on craq-sim: SrcA into the reduce is all-zero for reader1). Re-bind the tilize-reduce
            // unpack to THIS stick's actual input CB so reader1's in_cb_1 is the source for odd sticks.
#ifdef ARCH_QUASAR
            if constexpr (use_split_reader) {
                UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                    curr_in_cb_id, in_scalar_cb_id_0, tiles_to_reduce)));
            }
#endif
            MATH(WATCHER_RING_BUFFER_PUSH(0xC0FFEE11u));
            tile_regs_acquire();
            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
                UNPACK(WATCHER_RING_BUFFER_PUSH(0xC0FFEE12u));
                UNPACK(WATCHER_RING_BUFFER_PUSH((uint32_t)chunk));
                curr_in_cb.wait_front(1);
                unpack_tilizeA_B_block<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                    curr_in_cb_id,
                    curr_scalar_cb_id,
                    tiles_to_reduce,
                    0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/);
                for (uint32_t math_tile_idx = 0; math_tile_idx < tiles_to_reduce; ++math_tile_idx) {
                    reduce_tile_math<REDUCE_OP, REDUCE_DIM>(math_tile_idx, num_faces_in_input_tile);
                }
                curr_in_cb.pop_front(1);
            }
            tile_regs_commit();
            PACK(WATCHER_RING_BUFFER_PUSH(0xC0FFEE13u));
            tile_regs_wait();
            if constexpr (is_output_tiled) {
                // TILED output: accumulate sticks and perform tilization when needed.
                // pre_tilize_cb_id / fast_tilize_cb_id are only emitted under OUTPUT_TILED; even
                // though this branch is discarded by `if constexpr` on the non-tiled build, the
                // compiler still name-looks-up those DFB tokens, so the body must be #ifdef-gated.
#ifdef OUTPUT_TILED
                if (last_c_block) {
                    pack_untilize_dest<partial_iter_output_tiles>(pre_tilize_cb_id, 1, 0);
                    pre_tilize_cb.push_back(partial_iter_output_tiles);
                    tilize_stick_counter++;
                    tilize_stick_total++;
                } else {
                    pack_untilize_dest<max_tiles_per_iter>(pre_tilize_cb_id, 1, 0);
                    pre_tilize_cb.push_back(max_tiles_per_iter);
                }
                tile_regs_release();

                bool last_tile = num_out_sticks_this_core - tilize_stick_total < last_tile_height;
                if (tilize_stick_counter == TILE_HEIGHT || (last_tile && tilize_stick_counter == last_tile_height)) {
                    if (last_tile && last_tile_height != TILE_HEIGHT) {
                        pre_tilize_cb.wait_front(last_tile_height * in_ntiles_c);
                        // if the last tile is not whole we won't have pushed enough sticks, so we need to
                        // push some filler sticks to reach TILE_HEIGHT to make sure the CB pointers are correct
                        // before calling tilize
                        uint32_t filler_stick_tiles =
                            (TILE_HEIGHT - last_tile_height) *
                            ((in_nblocks_c - 1) * max_tiles_per_iter + partial_iter_output_tiles);
                        pre_tilize_cb.push_back(filler_stick_tiles);
                    }
                    PACK((pack_untilize_uninit(pre_tilize_cb_id)));

                    unpack_tilizeA_B_uninit(curr_in_cb_id);
                    pack_reconfig_data_format(out_cb_id);

                    // Hand the freshly-written L1 region off to the consumer view of the
                    // multi-format CB. pre_tilize_cb_id was pushed in TILE_HEIGHT*in_ntiles_c
                    // stick-pages (page_size = TILE_WIDTH*nbytes); fast_tilize_cb_id sees the
                    // same bytes as in_ntiles_c full tiles (page_size = tile_size). Both views
                    // advance by the same number of bytes per round so their rd/wr pointers
                    // stay aligned. The producer-view wait_front/pop_front/reserve_back below
                    // continues to drive the producer pointer ledger.
                    fast_tilize_cb.push_back(in_ntiles_c);
                    fast_tilize_cb.wait_front(in_ntiles_c);

#ifndef ARCH_QUASAR
                    fast_tilize_init(fast_tilize_cb_id, in_ntiles_c, out_cb_id);
                    fast_tilize_block(fast_tilize_cb_id, in_ntiles_c, out_cb_id);
                    fast_tilize_uninit(fast_tilize_cb_id, out_cb_id, in_ntiles_c);
#else
                    // QSR: fast_tilize is unported on Quasar (fast_tilize.h is #ifndef ARCH_QUASAR). Use the
                    // supported compute-API tilize on the same fast_tilize_cb view — all CB push/wait/pop
                    // plumbing above and below is preserved. NEEDS ON-QUASAR VALIDATION via the global
                    // avg_pool2d correctness test: the fast->regular tilize swap keeps CB sync intact, but
                    // the tile-view read semantics must be confirmed.
                    tilize_init(fast_tilize_cb_id, in_ntiles_c, out_cb_id);
                    tilize_block(fast_tilize_cb_id, in_ntiles_c, out_cb_id);
                    tilize_uninit(fast_tilize_cb_id, out_cb_id);
#endif

                    out_cb.push_back(in_ntiles_c);
                    fast_tilize_cb.pop_front(in_ntiles_c);
                    fast_tilize_cb.reserve_back(in_ntiles_c);
                    pre_tilize_cb.pop_front(TILE_HEIGHT * in_ntiles_c);
                    pre_tilize_cb.reserve_back(TILE_HEIGHT * in_ntiles_c);

                    tilize_stick_counter = 0;

                    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce)));
                    // init math for reduction again since FPU gets reprogrammed by tilize.
                    // Both WH and Quasar llk_math_reduce_init require the (operandA, operandB) CBs.
                    MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, DST_ACCUM_MODE, MATH_FIDELITY>(
                        in_cb_id_0, in_scalar_cb_id_0)));
#ifdef ARCH_BLACKHOLE
                    // need this on BH to set swizzle bit before pack untilize dest
                    MATH((llk_math_reconfig_remap(true)));
#endif

                    if constexpr (is_output_block_format) {
                        pack_reconfig_data_format(pre_tilize_cb_id);
                    }
#ifndef ARCH_QUASAR
                    PACK((llk_pack_untilize_init<max_tiles_per_iter, max_tiles_per_iter, false, false, TILE_C_DIM>(
                        pre_tilize_cb_id)));
#else
                    // QSR: Quasar llk_pack_untilize_init takes only <block_ct_dim, full_ct_dim>; use the
                    // compute-API pack_untilize_dest_init (as at the top of the kernel), which forwards 2 on Quasar.
                    pack_untilize_dest_init<max_tiles_per_iter>(pre_tilize_cb_id);
#endif
                }
#endif  // OUTPUT_TILED
            } else {
                // [DEBUG] Pack the reduced DEST and DPRINT the packed L1. PACK_TO_SCRATCH picks the CB:
                //   ==1: full-tile (32x32) pack into the scratch CB; out_cb (reserved above) gets a
                //        garbage balancing push so nothing waiting on it stalls.
                //   ==0: production RM path — narrow pack_untilize straight into out_cb (reserved above).
                // pack_cb is bound to whichever CB is active so the pack + DPRINT are written once.
#if PACK_TO_SCRATCH == 1
                // Produce the full-tile pack into THIS stick's reader scratch CB (reader0->scratch_cb_0,
                // reader1->scratch_cb_1 — same reader0 split as the input CB). The DM reader consumes +
                // DPRINTs it. NO compute-side consume. out_cb still gets a garbage balancing push so its
                // self-loop stays balanced.
#ifdef SPLIT_READER
                const uint32_t curr_scratch_cb_id = reader0 ? scratch_cb_id_0 : scratch_cb_id_1;
                DataflowBuffer curr_scratch_cb = reader0 ? scratch_cb_0 : scratch_cb_1;
#else
                const uint32_t curr_scratch_cb_id = scratch_cb_id_0;
                DataflowBuffer curr_scratch_cb = scratch_cb_0;
#endif
                // Reserve/push the WHOLE scratch CB (one full-tile write) so the single-tile scratch
                // serializes per stick: stick N+1 can't reserve until the DM reader pops stick N's whole
                // tile, so the full-tile pack never overlaps a still-unread tile.
                curr_scratch_cb.reserve_back(scratch_npages);
#ifdef ARCH_QUASAR
                // FIX (pack-side analog of the unpack rebind above): on Quasar, pack_untilize_dest(ocb) does
                // NOT re-derive the packer's OUTPUT buffer descriptor from the ocb passed at call time -- it
                // stays bound to whatever pack_untilize_dest_init set (done once with scratch_cb_0). So the
                // odd (reader1) stick, packed with scratch_cb_1, would write scratch_cb_0's L1 and leave
                // scratch_cb_1 unwritten -> reader1 reads zeros. Re-init the pack descriptor to THIS stick's
                // scratch CB so the odd stick actually lands in scratch_cb_1.
                if constexpr (use_split_reader) {
                    pack_untilize_dest_init<max_tiles_per_iter>(curr_scratch_cb_id);
                }
#endif
#if DEBUG_PRINT == 1
                // Which scratch CB and where does compute pack THIS stick? Compare wptr to the reader's
                // rdptr: same address + reader reads 0 => reduce produced 0 (input); differ => routing.
                PACK(DPRINT(
                    "PACKW n={} reader0={} scr_cb={} wptr={}\n",
                    (uint32_t)n,
                    (uint32_t)reader0,
                    (uint32_t)curr_scratch_cb_id,
                    (uint32_t)curr_scratch_cb.get_write_ptr()));
#endif
                if (last_c_block) {
                    pack_untilize_dest<partial_iter_output_tiles>(curr_scratch_cb_id, 1, 0);
                } else {
                    pack_untilize_dest<max_tiles_per_iter>(curr_scratch_cb_id, 1, 0);
                }
                tile_regs_release();
                curr_scratch_cb.push_back(scratch_npages);  // hand off to the DM reader, which writes the output
#else
                // Production RM path: narrow pack straight into out_cb (already reserved above).
                if (last_c_block) {
                    pack_untilize_dest<partial_iter_output_tiles>(out_cb_id, 1, 0);
                } else {
                    pack_untilize_dest<max_tiles_per_iter>(out_cb_id, 1, 0);
                }
                tile_regs_release();
                out_cb.push_back(output_faces);
#endif
#if DEBUG_PRINT == 1
                PACK(DPRINT(
                    "PACKDBG n={} c_i={} ofaces={} scratch={}\n",
                    (uint32_t)n,
                    (uint32_t)c_i,
                    (uint32_t)output_faces,
                    (uint32_t)PACK_TO_SCRATCH));
#endif
            }
        }
        if constexpr (!one_scalar_per_core) {
            curr_scalar_cb.pop_front(1);
        }
    }
}
