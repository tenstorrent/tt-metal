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
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

//   0 = production path: narrow pack_untilize straight into the real output CB (out_cb), then DPRINT it.
//   1 = experiment:      full-tile (32x32) pack of the reduced DEST into the scratch CB, then DPRINT it
//                        (out_cb still gets a balancing push with garbage).
// The pack_untilize_dest_init and the pack call all follow this switch, so flipping this one value moves
// the whole pack pipeline between the two CBs consistently.
// NOTE: PACK_TO_SCRATCH==1 requires the reader-side scratch consume (reader_pool_2d.cpp) to be enabled
// too — compute PRODUCES the scratch CB and the DM reader CONSUMES it; they must be on/off together.
#define PACK_TO_SCRATCH 1

#define ALWI inline __attribute__((always_inline))

#define FACE_HEIGHT 16
#define FACE_WIDTH 16
#define TILE_HEIGHT 32
#define TILE_WIDTH 32

// out_cb is the borrowed output shard and compute is its only producer and consumer (self-loop): the result
// stays resident, so pop each block right after it is pushed to leave the buffer balanced. dummy_unpack
// orders the pop after the wait on Quasar (nothing unpacks this buffer, and a bare wait_front -> pop_front
// traps the unpacker).
template <typename DFB>
ALWI void release_out_pages(DFB& out_cb, uint32_t num_pages) {
    out_cb.wait_front(num_pages);
    dummy_unpack(out_cb.get_id());
    out_cb.pop_front(num_pages);
}

void kernel_main() {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet. When ntiles_hw > 1 the large
    // kernel is called
    constexpr uint32_t in_ntiles_c = get_arg(args::in_ntiles_c);
    constexpr uint32_t window_size_hw = get_arg(args::window_size_hw);
    constexpr uint32_t scratch_npages = get_arg(args::scratch_npages);

    constexpr uint32_t max_out_sticks_per_core = get_arg(args::max_out_sticks_per_core);
    constexpr uint32_t in_c = get_arg(args::in_c);
    constexpr uint32_t in_nblocks_c = get_arg(args::in_nblocks_c);
    constexpr uint32_t max_sticks_for_reduction = get_arg(args::max_sticks_for_reduction);

    // CB ids are Metal 2.0 DFB tokens. Keep the legacy variable names so the rest of the
    // kernel (DataflowBuffer construction and LLK calls taking a uint32_t CB id) is unchanged;
    // dfb::<name> converts implicitly to uint32_t.
    constexpr auto in_cb_id_0 = dfb::in_cb_0;
    constexpr auto in_scalar_cb_id_0 = dfb::in_scalar_cb_0;
    constexpr auto out_cb_id = dfb::out_cb;
    constexpr auto scratch_cb_id_0 = dfb::scratch_cb_0;
    constexpr bool one_scalar_per_core = get_arg(args::one_scalar_per_core);
    constexpr bool is_output_tiled = get_arg(args::is_output_tiled);  // 1 = TILED, 0 = ROW_MAJOR
    constexpr bool is_output_block_format = (bool)get_arg(args::is_output_block_format);
#ifdef OUTPUT_TILED
    constexpr auto pre_tilize_cb_id = dfb::pre_tilize_cb;
#endif

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
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    // The host factory resolves DEST-capacity limits (fp32 vs fp16, half-sync vs full-sync) and
    // equal-width c-block constraints into a single value; kernels consume it directly.
    constexpr uint32_t MAX_TILES_PER_REDUCTION = get_arg(args::max_tiles_per_reduction);
    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;
    static_assert(partial_iter_output_tiles == max_tiles_per_iter, "c-blocks must all be the same width");

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
    DataflowBuffer out_cb(out_cb_id);
    DataflowBuffer scratch_cb_0(scratch_cb_id_0);
#ifdef OUTPUT_TILED
    DataflowBuffer pre_tilize_cb(pre_tilize_cb_id);
    auto& pack_target_cb = pre_tilize_cb;
#else
    auto& pack_target_cb = scratch_cb_0;
#endif

    // Packer dest is baked at init (hw_startup + pack_untilize_dest_init) and must match the CB
    // the per-stick loop packs into. tilizeA_B_reduce_init only programs unpack+math (no ocb).
#if PACK_TO_SCRATCH == 1
    constexpr uint32_t pack_target_cb_id = is_output_tiled ? tilize_untilize_cb : scratch_cb_id_0;
#else
    constexpr uint32_t pack_target_cb_id = tilize_untilize_cb;
#endif
    compute_kernel_hw_startup(in_cb_id_0, in_scalar_cb_id_0, pack_target_cb_id);
    tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(in_cb_id_0, in_scalar_cb_id_0, max_tiles_per_iter);

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
    const uint32_t num_out_sticks_this_cluster =
        max_out_sticks_per_core ? max_out_sticks_per_core : get_arg(args::out_nhw_this_core);
    // This thread's sticks: the reader deals sticks round-robin in units of deal_sticks (unit u -> thread
    // u % T; one stick for row-major output, one 32-stick tile row for TILE output), so thread t owns
    // quotient + 1 units for t < units % T, and only the last unit can be partial — no divisibility
    // required. get_num_threads()/get_my_thread_id() are 1/0 off Quasar.
    constexpr uint32_t deal_sticks = is_output_tiled ? TILE_HEIGHT : 1;
    const uint32_t num_threads = get_num_threads();
    const uint32_t num_units = (num_out_sticks_this_cluster + deal_sticks - 1) / deal_sticks;
    const uint32_t num_units_this_thread =
        num_units / num_threads + (get_my_thread_id() < num_units % num_threads ? 1u : 0u);
    uint32_t num_out_sticks_per_thread = num_units_this_thread * deal_sticks;
    if (num_units_this_thread > 0 && (num_units - 1) % num_threads == get_my_thread_id()) {
        num_out_sticks_per_thread -= num_units * deal_sticks - num_out_sticks_this_cluster;
    }

    for (uint32_t n = 0; n < num_out_sticks_per_thread; ++n) {
        const uint32_t curr_scalar_cb_id = in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = in_cb_id_0;
        DataflowBuffer curr_scalar_cb = in_scalar_cb_0;
        DataflowBuffer curr_in_cb = in_cb_0;
        if constexpr (!one_scalar_per_core) {
            curr_scalar_cb.wait_front(1);
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
#if PACK_TO_SCRATCH == 0
            if constexpr (!is_output_tiled) {
                out_cb.reserve_back(output_faces);
            }
#endif
            // Re-init the fused tilize+reduce for THIS stick/c-block through the compute API rather than
            // hand-issuing individual UNPACK/MATH llk_* calls: tiles_to_reduce changes across c-blocks
            // (e.g. 4 then 2 for 6 tiles / 192c), so UNPACK and MATH must both be re-programmed for the
            // new count (PACK is re-init'd via pack_untilize_dest_init below).
            tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
                curr_in_cb_id, curr_scalar_cb_id, tiles_to_reduce);
            tile_regs_acquire();
            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
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
            tile_regs_wait();
#if PACK_TO_SCRATCH == 1
            // Each output stick is packed as a full untilized 32x32 tile whose row 0 is the result (a narrow
            // 1x32 pack would read DEST in the tiny-tile layout, not the reduce's full-tile layout). Row-major
            // output: row 0 of a per-stick scratch entry, which the reader copies to the output. TILE output:
            // row n % 32 of this thread's tile-row entry; sticks go in ascending rows, so each overwrites the
            // previous stick's junk rows (the entry has 32 spare rows for the last stick's).
            const uint32_t tile_row = is_output_tiled ? n % TILE_HEIGHT : 0;
            if (first_c_block && tile_row == 0) {
                pack_target_cb.reserve_back(1);
            }
            pack_untilize_dest_init<max_tiles_per_iter, in_ntiles_c>(pack_target_cb_id);
            pack_untilize_dest<max_tiles_per_iter, in_ntiles_c>(pack_target_cb_id, 1, tile_row * in_nblocks_c + c_i);
            tile_regs_release();
            if (last_c_block) {
                if constexpr (!is_output_tiled) {
                    scratch_cb_0.push_back(1);  // hand off to the DM reader, which writes the output
                }
#ifdef OUTPUT_TILED
                if (tile_row == TILE_HEIGHT - 1 || n == num_out_sticks_per_thread - 1) {
                    // Tile row done (or this thread's last, partial one): tilize it into in_ntiles_c scratch
                    // tiles, which the reader copies to the output.
                    pre_tilize_cb.push_back(1);
                    pre_tilize_cb.wait_front(1);
                    pack_untilize_uninit(pre_tilize_cb_id);
                    unpack_tilizeA_B_uninit(curr_in_cb_id);
                    pack_reconfig_data_format(scratch_cb_id_0);
                    scratch_cb_0.reserve_back(in_ntiles_c);
#ifdef ARCH_QUASAR
                    // Quasar tilize_init does not program PACK; retarget from pack-untilize (pre_tilize) to scratch.
                    pack_init(scratch_cb_id_0);
#endif
                    fast_tilize_init(pre_tilize_cb_id, in_ntiles_c, scratch_cb_id_0);
                    fast_tilize_block(pre_tilize_cb_id, in_ntiles_c, scratch_cb_id_0);
                    fast_tilize_uninit(pre_tilize_cb_id, scratch_cb_id_0, in_ntiles_c);
                    scratch_cb_0.push_back(in_ntiles_c);
                    pre_tilize_cb.pop_front(1);

                    // Re-init fused tilize+reduce: tilize_init reprogrammed the FPU (A2D datacopy).
                    tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
                        in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce);
                    if constexpr (is_output_block_format) {
                        pack_reconfig_data_format(pre_tilize_cb_id);
                    }
                }
#endif  // OUTPUT_TILED
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
                release_out_pages(out_cb, output_faces);
#endif
        }
        if constexpr (!one_scalar_per_core) {
            curr_scalar_cb.pop_front(1);
        }
    }
    if constexpr (one_scalar_per_core) {
        // The single scalar tile was waited once up front and reused for every stick; pop it now so the
        // buffer is left balanced. dummy_unpack orders the pop after the wait on Quasar when no stick was
        // processed (a bare wait_front -> pop_front traps the unpacker).
        dummy_unpack(in_scalar_cb_id_0);
        in_scalar_cb_0.pop_front(1);
    }
}
