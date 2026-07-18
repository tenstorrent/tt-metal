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
    // target the same CB the per-stick loop actually packs into, or the Quasar packer (which bakes its
    // destination L1 address at init and ignores the runtime `ocb` arg thereafter -- the same class of
    // bug fixed in the halo op's pack_untilize, see pack_untilize.cpp) writes to the wrong place.
    // PACK_TO_SCRATCH's scratch-CB workaround only exists for the ROW_MAJOR output path (the `else`
    // branch below, which packs into curr_scratch_cb); the TILED output path (is_output_tiled) packs
    // straight into pre_tilize_cb (== tilize_untilize_cb) and never touches scratch_cb_0 at all, so the
    // init here must follow suit -- init'ing against scratch_cb_0 unconditionally left the TILED path's
    // packer permanently mis-targeted, and its downstream reader (which always waits on scratch_cb
    // regardless of is_output_tiled) never received the pushes it was waiting for -> reader deadlock.
#if PACK_TO_SCRATCH == 1
    // Both scratch CBs share the same full-tile geometry, so init once with scratch_cb_0.
    constexpr uint32_t pack_target_cb_id = is_output_tiled ? tilize_untilize_cb : scratch_cb_id_0;
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
            // Re-init the fused tilize+reduce for THIS stick/c-block through the compute API rather than
            // hand-issuing individual UNPACK/MATH llk_* calls. This re-programs UNPACK and MATH together, which
            // is required because both change per iteration:
            //   (a) split-reader: even sticks read in_cb_0, odd sticks read in_cb_1 -- the unpack-tilize
            //       descriptor must re-bind to THIS stick's input CB, else reader1 re-reduces reader0's window;
            //   (b) tiles_to_reduce changes across c-blocks (e.g. 4 then 2 for 6 tiles / 192c) -- UNPACK and
            //       MATH must both be re-programmed for the new count (PACK is re-init'd via
            //       pack_untilize_dest_init below).
            // Use the *_short variant: the full tilizeA_B_reduce_init (called once at kernel start) also runs
            // llk_*_hw_configure, and re-running hw_configure per c-block corrupts unpacker state (UNPACKER
            // fault). Re-initing only some engines desynced them (Risc IB interrupt, watcher 0x19, on MATH then
            // PACK); the short compute API keeps unpack+math in lockstep without the illegal per-block reconfig.
            tilizeA_B_reduce_init_short<neginf_srca_maxpool, zero_srca_avgpool>(
                curr_in_cb_id, curr_scalar_cb_id, tiles_to_reduce, pack_target_cb_id);
            MATH(WATCHER_RING_BUFFER_PUSH(0xC0FFEE11u));
            tile_regs_acquire();
            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
                UNPACK(WATCHER_RING_BUFFER_PUSH(0xC0FFEE12u));
                UNPACK(WATCHER_RING_BUFFER_PUSH((uint32_t)chunk));
                curr_in_cb.wait_front(1);
                // [DIAG] reduce-input geometry (Bug 1 straddle check): rd = strided-read base, esz = bytes
                // per entry, t2r = tiles this reduce strides over. If the strided row walk for t2r tiles
                // exceeds esz it reads into the next entry (wrong rows). First few sticks only (flood guard).
                if (n < 4) {
                    UNPACK(DPRINT(
                        "POOLRED n={} chunk={} cb={} rd={} esz={} nent={} t2r={} intiles={}\n",
                        (uint32_t)n,
                        (uint32_t)chunk,
                        (uint32_t)curr_in_cb_id,
                        (uint32_t)curr_in_cb.get_read_ptr(),
                        (uint32_t)curr_in_cb.get_entry_size(),
                        (uint32_t)curr_in_cb.get_total_num_entries(),
                        (uint32_t)tiles_to_reduce,
                        (uint32_t)in_ntiles_c));
                }
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
                    // plumbing above and below is preserved.
                    //
                    // QSR packer retarget (same defect class as the halo pack_untilize fix and the
                    // OUTPUT_TILED deadlock fix above): Quasar's `tilize_init` (tilize.h) only programs
                    // UNPACK+MATH -- unlike WH/BH it does NOT call a PACK-side init, because on Quasar the
                    // packer's destination CB/tensor-shape descriptor is baked into tensix state by an
                    // explicit init call and is NOT reprogrammed implicitly (see reconfig_data_format.h's
                    // ARCH_QUASAR note: "When the pack output operand changes, call pack_init(new_cb_id)
                    // before pack_tile"). Immediately before this point the packer was left in
                    // pack-untilize mode targeting pre_tilize_cb_id (from the per-stick reduce loop's
                    // pack_untilize_dest_init below / at kernel entry). `pack_reconfig_data_format` above
                    // only reprograms the THCON data-format (gasket), not the pack MOP/descriptor, so
                    // without an explicit `llk_pack_init(out_cb_id)` here, `tilize_block`'s Quasar-path
                    // `llk_pack<out_of_order>(...)` call packs through a descriptor still bound to
                    // pre_tilize_cb_id in untilize mode -- the real out_cb never receives the tilized tile
                    // (observed as PCC 0.0 / all-zero output on the OUTPUT_TILED avg-pool path).
                    PACK((llk_pack_init(out_cb_id)));
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
                // Model A (wide-reduction fix): the scratch CB holds ONE contiguous full-width (in_ntiles_c)
                // output stick, so reserve it ONCE per stick (on the first c-block) -- NOT per c-block.
                // Reserving/pushing a full stick per c-block advanced wr_entry_idx mid-stick, overran the
                // 2-slot ring, and grew the packer's L1 base (base_l1 = wr_entry_idx * ...) past the buffer-
                // descriptor limit -> Risc IB interrupt (0x19), with the fault address creeping run-to-run as
                // base_l1 grew. Each c-block packs its channel slice into this shared stick below; the whole
                // stick is pushed once on the last c-block so the DM reader consumes exactly one stick per pop.
                if (first_c_block) {
                    curr_scratch_cb.reserve_back(scratch_npages);
                }
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
                // QSR fix (split-reader second stream): the top-of-kernel pack_untilize_dest_init targets
                // scratch_cb_0 only, so odd (reader1) sticks packing into scratch_cb_1 used a packer
                // descriptor still bound to scratch_cb_0 -> the pack landed nowhere and reader1 read an
                // all-zero tile. Re-init the pack_untilize for THIS stick's scratch CB before packing so
                // both scratch_cb_0 (even) and scratch_cb_1 (odd) get a valid, CB-matched descriptor.
                // [DIAG] pack bounds (Bug 2 PACR0_TILE_INC 0x19): ntiles = tiles this pack_untilize_dest
                // writes; cap = scratch CB byte capacity; npages/esz = its entry geometry. If
                // ntiles*esz > cap (or ntiles exceeds what the scratch descriptor addresses) the pack tile
                // increment crosses L1_LIMIT_ADDR -> fault. Printed BEFORE the pack so it flushes pre-fault.
                if (n < 4) {
                    PACK(DPRINT(
                        "POOLPACK n={} scr_cb={} wptr={} cap={} npages={} esz={} ntiles={} intiles={}\n",
                        (uint32_t)n,
                        (uint32_t)curr_scratch_cb_id,
                        (uint32_t)curr_scratch_cb.get_write_ptr(),
                        (uint32_t)(curr_scratch_cb.get_total_num_entries() * curr_scratch_cb.get_entry_size()),
                        (uint32_t)scratch_npages,
                        (uint32_t)curr_scratch_cb.get_entry_size(),
                        (uint32_t)(last_c_block ? partial_iter_output_tiles : max_tiles_per_iter),
                        (uint32_t)in_ntiles_c));
                }
                // Pack this c-block into its channel slice of the shared full-width stick. full_ct_dim =
                // in_ntiles_c makes the untilize row stride span the whole in_ntiles_c-tile-wide stick
                // (llk_pack_untilize stride_offset_0 = num_faces_c_dim * FULL_CT_DIM); block_c_index places the
                // slice so the c-blocks tile contiguously (l1 tile offset = block_c_index * block_ct_dim):
                //   c0 -> block_c_index 0            -> tiles 0..3
                //   c1 -> block_c_index 4/2 = 2      -> tiles 4..5
                // Init width must equal pack width per c-block (pack_untilize.h contract), so init per c-block.
                if (last_c_block) {
                    pack_untilize_dest_init<partial_iter_output_tiles, in_ntiles_c>(curr_scratch_cb_id);
                    pack_untilize_dest<partial_iter_output_tiles, in_ntiles_c>(
                        curr_scratch_cb_id, 1, (c_i * max_tiles_per_iter) / partial_iter_output_tiles);
                } else {
                    pack_untilize_dest_init<max_tiles_per_iter, in_ntiles_c>(curr_scratch_cb_id);
                    pack_untilize_dest<max_tiles_per_iter, in_ntiles_c>(curr_scratch_cb_id, 1, c_i);
                }
                tile_regs_release();
                if (last_c_block) {
                    // Push the whole stick once, after every c-block has packed its slice, so the DM reader
                    // consumes exactly one full stick per wait_front/pop_front(scratch_npages) -- the previous
                    // per-c-block push overran the ring (see the reserve note above).
                    //
                    // [DEBUG scratch scaffold] Producer-side pack-write commit barrier. push_back() below posts
                    // the SPSC credit the DM reader spins on (reader_pool_2d.cpp wait_front), after which it reads
                    // this stick's row DIRECTLY from TL1. On HW the credit instruction itself waits for the packer
                    // write to drain (TT_PUSH_TILES packer_wr_done_wait_mask=0x1 in llk_push_tiles), but the Quasar
                    // emulator does NOT honor that embedded sub-field, so the counter can post before
                    // pack_untilize_dest's TL1 write lands -> the reader reads a stale/empty scratch slot (a
                    // fraction of sticks dropped->0 or dup->neighbor: PCC 0.897; watcher latency hides it, since
                    // the scratch CB is single-buffered so the credit is the only producer->consumer serializer).
                    // Drain the packer engine before posting the credit. Mirrors the "wait for pack to finish"
                    // STALLWAIT idiom in llk_pack_common.h:307. No-op-equivalent on HW (redundant with the wr_done
                    // wait); remove once the sim models PUSH_TILES.packer_wr_done_wait_mask.
                    // Quasar-only: this barrier fixes the emulator's failure to honor PUSH_TILES'
                    // packer_wr_done_wait_mask (see comment above). WH/BH HW honor it, so the stall is a no-op
                    // there -- and TTI_STALLWAIT has a DIFFERENT arity per arch (Quasar takes 4 args, WH takes 2),
                    // so it must be arch-gated to compile at all. NB: TTI_STALLWAIT expands to a bare __asm__
                    // statement, so it must NOT be wrapped in the expression-form PACK((...)) parens -- PACK(...)
                    // is variadic, so the comma-separated p_stall args pass straight through as one asm statement.
#ifdef ARCH_QUASAR
                    PACK(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::NOTHING, p_stall::PACK));
#endif
                    curr_scratch_cb.push_back(scratch_npages);  // hand off to the DM reader, which writes the output
                }
#else
                // Production RM path: narrow pack straight into out_cb (already reserved above). Pair the
                // pack-untilize init with a matching width per c-block (same contract as the scratch path).
                if (last_c_block) {
                    pack_untilize_dest_init<partial_iter_output_tiles>(out_cb_id);
                    pack_untilize_dest<partial_iter_output_tiles>(out_cb_id, 1, 0);
                } else {
                    pack_untilize_dest_init<max_tiles_per_iter>(out_cb_id);
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
