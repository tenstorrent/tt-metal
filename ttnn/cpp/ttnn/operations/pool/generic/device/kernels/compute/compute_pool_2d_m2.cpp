// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of compute_pool_2d.cpp. Forked (not edited in place) because compute_pool_2d.cpp is
// also used by the rotate and grid_sample factories; only the Pool2D::MultiCore factory's compute
// kernel moves to the Metal 2.0 named-binding form. Logic is identical to compute_pool_2d.cpp; only
// the access mechanism changes: CB-id CTAs -> dfb::<name>, scalar/dimension CTAs -> get_arg(args::),
// the positional runtime arg -> get_arg(args::out_nhw_this_core).

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
#include "experimental/kernel_args.h"

#define DEBUG_PRINT 0

#if DEBUG_PRINT == 1
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#include "api/debug/dprint_tensix.h"
#include "tools/profiler/kernel_profiler.hpp"
#endif

#define ALWI inline __attribute__((always_inline))

#define FACE_HEIGHT 16
#define FACE_WIDTH 16
#define TILE_HEIGHT 32
#define TILE_WIDTH 32

void kernel_main() {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet. When ntiles_hw > 1 the large
    // kernel is called
    constexpr uint32_t in_ntiles_c = get_arg(args::in_ntiles_c);
    constexpr uint32_t window_size_hw = get_arg(args::window_size_hw);

    constexpr uint32_t split_reader = get_arg(args::split_reader);

    constexpr uint32_t max_out_sticks_per_core = get_arg(args::max_out_sticks_per_core);
    constexpr uint32_t in_c = get_arg(args::in_c);
    constexpr uint32_t in_nblocks_c = get_arg(args::in_nblocks_c);
    constexpr uint32_t max_sticks_for_reduction = get_arg(args::max_sticks_for_reduction);

    constexpr uint32_t in_cb_id_0 = dfb::in_0;
#ifdef SPLIT_READER
    constexpr uint32_t in_cb_id_1 = dfb::in_1;  // for split reader
#endif
    constexpr uint32_t in_scalar_cb_id_0 = dfb::in_scalar_0;
#ifdef HAS_SECOND_SCALAR_CB
    constexpr uint32_t in_scalar_cb_id_1 = dfb::in_scalar_1;
#endif
    constexpr uint32_t out_cb_id = dfb::out;
    constexpr bool one_scalar_per_core = get_arg(args::one_scalar_per_core);
#ifdef IS_OUTPUT_TILED
    constexpr uint32_t pre_tilize_cb_id = dfb::pre_tilize;
#endif
    constexpr bool is_output_tiled = get_arg(args::is_output_tiled);  // 1 = TILED, 0 = ROW_MAJOR
    constexpr bool is_output_block_format = (bool)get_arg(args::is_output_block_format);
    // fast_tilize_cb_id is a consumer-view alias of pre_tilize_cb_id (same L1 region,
    // full-tile face_geometry = {face_r_dim=16, num_faces=4}). Used as the input operand
    // to fast_tilize so the unpacker/math read the correct face count from CB metadata.
#ifdef IS_OUTPUT_TILED
    constexpr uint32_t fast_tilize_cb_id = dfb::fast_tilize;
#endif

    constexpr bool use_split_reader = split_reader;

    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    constexpr uint32_t num_faces_in_input_tile =
        (max_sticks_for_reduction < TILE_HEIGHT || window_size_hw <= FACE_HEIGHT) ? 2 : 4;
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

    // tilize_untilize_cb selects the pre_tilize view for TILED output, else the output CB. When the
    // op is not tiled, dfb::pre_tilize is unbound (no IS_OUTPUT_TILED define), so the ternary that
    // referenced pre_tilize_cb_id is #ifdef-gated to the bound branch.
#ifdef IS_OUTPUT_TILED
    constexpr uint32_t tilize_untilize_cb = is_output_tiled ? pre_tilize_cb_id : out_cb_id;
#else
    constexpr uint32_t tilize_untilize_cb = out_cb_id;
#endif

    experimental::CB in_scalar_cb_0(in_scalar_cb_id_0);
#ifdef HAS_SECOND_SCALAR_CB
    experimental::CB in_scalar_cb_1(in_scalar_cb_id_1);
#endif
    experimental::CB in_cb_0(in_cb_id_0);
#ifdef SPLIT_READER
    experimental::CB in_cb_1(in_cb_id_1);
#endif
    experimental::CB out_cb(out_cb_id);
#ifdef IS_OUTPUT_TILED
    experimental::CB pre_tilize_cb(pre_tilize_cb_id);
    experimental::CB fast_tilize_cb(fast_tilize_cb_id);
#endif

    tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
        in_cb_id_0, in_scalar_cb_id_0, max_tiles_per_iter, tilize_untilize_cb);

    pack_untilize_dest_init<max_tiles_per_iter>(tilize_untilize_cb);

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
    for (uint32_t n = 0; n < num_out_sticks_this_core; ++n) {
        const bool reader0 = !(use_split_reader && (n & 0x1));
        const bool use_reader1_scalar = !reader0 && !one_scalar_per_core;
        // The "reader1" operands (in_scalar_1 / in_1) only exist when their DFB is bound, i.e. when
        // split-reader (and, for the scalar, the second-scalar) DFB is present. When unbound the
        // ternary's reader1 branch can never be taken at runtime (use_reader1_scalar / !reader0 are
        // always false without split-reader), so #ifdef-select the reader0 operand directly.
#ifdef HAS_SECOND_SCALAR_CB
        const uint32_t curr_scalar_cb_id = use_reader1_scalar ? in_scalar_cb_id_1 : in_scalar_cb_id_0;
        experimental::CB curr_scalar_cb = use_reader1_scalar ? in_scalar_cb_1 : in_scalar_cb_0;
#else
        const uint32_t curr_scalar_cb_id = in_scalar_cb_id_0;
        experimental::CB curr_scalar_cb = in_scalar_cb_0;
#endif
#ifdef SPLIT_READER
        const uint32_t curr_in_cb_id = !reader0 ? in_cb_id_1 : in_cb_id_0;
        experimental::CB curr_in_cb = reader0 ? in_cb_0 : in_cb_1;
#else
        const uint32_t curr_in_cb_id = in_cb_id_0;
        experimental::CB curr_in_cb = in_cb_0;
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
            if constexpr (!is_output_tiled) {
                out_cb.reserve_back(output_faces);
            }
            if constexpr (tilize_reconfig) {
                if (first_c_block || last_c_block) {
                    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce)));
                }
            }
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
            if constexpr (is_output_tiled) {
                // TILED output: accumulate sticks and perform tilization when needed.
                // dfb::pre_tilize / dfb::fast_tilize are only bound on the tiled path (IS_OUTPUT_TILED),
                // which is exactly when is_output_tiled is true; #ifdef-guard the body so the unbound
                // tokens never enter name lookup on the row-major build.
#ifdef IS_OUTPUT_TILED
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

                    fast_tilize_init(fast_tilize_cb_id, in_ntiles_c, out_cb_id);
                    fast_tilize_block(fast_tilize_cb_id, in_ntiles_c, out_cb_id);
                    fast_tilize_uninit(fast_tilize_cb_id, out_cb_id, in_ntiles_c);

                    out_cb.push_back(in_ntiles_c);
                    fast_tilize_cb.pop_front(in_ntiles_c);
                    fast_tilize_cb.reserve_back(in_ntiles_c);
                    pre_tilize_cb.pop_front(TILE_HEIGHT * in_ntiles_c);
                    pre_tilize_cb.reserve_back(TILE_HEIGHT * in_ntiles_c);

                    tilize_stick_counter = 0;

                    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce)));
                    // init math for reduction again since FPU gets reprogrammed by tilize
                    MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, DST_ACCUM_MODE, MATH_FIDELITY>()));
#ifdef ARCH_BLACKHOLE
                    // need this on BH to set swizzle bit before pack untilize dest
                    MATH((llk_math_reconfig_remap(true)));
#endif

                    if constexpr (is_output_block_format) {
                        pack_reconfig_data_format(pre_tilize_cb_id);
                    }
                    PACK((llk_pack_untilize_init<max_tiles_per_iter, max_tiles_per_iter, false, false, TILE_C_DIM>(
                        pre_tilize_cb_id)));
                }
#endif  // IS_OUTPUT_TILED
            } else {
                // ROW_MAJOR output: pack directly to output CB
                if (last_c_block) {
                    pack_untilize_dest<partial_iter_output_tiles>(out_cb_id, 1, 0);
                } else {
                    pack_untilize_dest<max_tiles_per_iter>(out_cb_id, 1, 0);
                }
                out_cb.push_back(output_faces);
                tile_regs_release();
            }
        }
        if constexpr (!one_scalar_per_core) {
            curr_scalar_cb.pop_front(1);
        }
    }
}
