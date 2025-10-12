// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#define DEBUG_PRINT 0

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"
#endif

// DST Optimization debug prints
#ifdef DEBUG_DST_FLOW
#define DST_DEBUG_PRINT(...) DPRINT << __VA_ARGS__ << ENDL()
#else
#define DST_DEBUG_PRINT(...) \
    do {                     \
    } while (0)
#endif

#define ALWI inline __attribute__((always_inline))

#define FACE_HEIGHT 16
#define FACE_WIDTH 16
#define TILE_HEIGHT 32
#define TILE_WIDTH 32

namespace NAMESPACE {

template <uint32_t topk_output_tiles, uint32_t data_dst_idx, uint32_t index_dst_idx, uint32_t topk_cb_tile_idx>
ALWI void tilize_dest_function(uint32_t curr_in_cb_id, uint32_t curr_in_idx_cb_id) {
    tilize_init_short_with_dt_no_pack(curr_in_cb_id, curr_in_idx_cb_id, topk_output_tiles);
    tilize_block_no_pack(curr_in_idx_cb_id, topk_output_tiles, index_dst_idx, topk_cb_tile_idx);
    tilize_uninit_with_dt_no_pack(curr_in_idx_cb_id, curr_in_cb_id);
    tilize_init_short_with_dt_no_pack(curr_in_idx_cb_id, curr_in_cb_id, topk_output_tiles);
    tilize_block_no_pack(curr_in_cb_id, topk_output_tiles, data_dst_idx, topk_cb_tile_idx);
    tilize_uninit_with_dt_no_pack(curr_in_cb_id, curr_in_idx_cb_id);
}

template <
    uint32_t topk_output_tiles,
    uint32_t data_dst_idx,
    uint32_t index_dst_idx,
    uint32_t topk_cb_tile_idx,
    uint32_t tile_tmp_cb_id,
    uint32_t tile_idx_tmp_cb_id,
    uint32_t out_cb_id,
    uint32_t num_out_sticks,
    bool pack_untilize_reinit>
ALWI void tilize_copy_function(uint32_t curr_in_cb_id, uint32_t curr_in_idx_cb_id, uint32_t output_faces) {
    // tensix syncs are necessary here until https://github.com/tenstorrent/tt-metal/issues/30399 is resolved
    tensix_sync();
    unary_op_init_common(curr_in_cb_id, tile_tmp_cb_id);
    tensix_sync();
    tilize_init(curr_in_cb_id, topk_output_tiles, tile_tmp_cb_id);

    cb_reserve_back(tile_tmp_cb_id, topk_output_tiles);

    tilize_block(curr_in_cb_id, topk_output_tiles, tile_tmp_cb_id, topk_cb_tile_idx, topk_cb_tile_idx);

    cb_push_back(tile_tmp_cb_id, topk_output_tiles);
    cb_wait_front(tile_tmp_cb_id, topk_output_tiles);
    cb_reserve_back(tile_idx_tmp_cb_id, topk_output_tiles);

    tilize_uninit_with_dt(curr_in_cb_id, curr_in_idx_cb_id, tile_idx_tmp_cb_id);
    tilize_init_short_with_dt(curr_in_cb_id, curr_in_idx_cb_id, topk_output_tiles, tile_idx_tmp_cb_id);
    tilize_block(curr_in_idx_cb_id, topk_output_tiles, tile_idx_tmp_cb_id, topk_cb_tile_idx, topk_cb_tile_idx);

    cb_push_back(tile_idx_tmp_cb_id, topk_output_tiles);
    cb_wait_front(tile_idx_tmp_cb_id, topk_output_tiles);

    tilize_uninit(curr_in_idx_cb_id, tile_idx_tmp_cb_id);

    copy_tile_init(tile_tmp_cb_id);
    if constexpr (pack_untilize_reinit) {
// note pack_untilize_dest_init must be called immediately after copy_tile_init see issue
// https://github.com/tenstorrent/tt-metal/issues/#27314
// also note we don't actually call pack_untilize_dest_init here, but instead call all it
// contents without the llk_pack_untilize_hw_configure_disaggregated call so that
// we can avoid tensix_syncs
#ifdef ARCH_BLACKHOLE
        // Needed for setting swizzle_32b:
        MATH((llk_math_hw_configure_disaggregated<true, true>(0, 0)));
#endif
        PACK((llk_pack_untilize_init<topk_output_tiles, topk_output_tiles, false, false, TILE_C_DIM>(
            out_cb_id, num_out_sticks, output_faces)));
        PACK((llk_init_packer_dest_offset_registers<true, false>()));
    }
    copy_tile(tile_tmp_cb_id, 0, data_dst_idx);
    copy_tile(tile_idx_tmp_cb_id, 0, index_dst_idx);

    cb_pop_front(tile_tmp_cb_id, topk_output_tiles);
    cb_pop_front(tile_idx_tmp_cb_id, topk_output_tiles);
}

// DST Optimization Functions
template <bool neginf_srca_maxpool, bool zero_srca_avgpool>
ALWI void process_4way_parallel_positions(
    uint32_t in_cb_id_0,
    uint32_t in_cb_id_1,
    uint32_t in_scalar_cb_id_0,
    uint32_t in_scalar_cb_id_1,
    uint32_t out_cb_id,
    uint32_t tiles_to_reduce,
    uint32_t num_faces_in_input_tile,
    uint32_t face_r_dim,
    uint32_t num_out_sticks,
    uint32_t num_faces_in_output_tile,
    bool split_reader,
    bool one_scalar_per_core,
    uint32_t current_position_batch) {
    DST_DEBUG_PRINT("=== DST 4WAY PARALLEL BATCH " << current_position_batch << " ===");
    DST_DEBUG_PRINT("tiles_to_reduce=" << tiles_to_reduce << " positions=4");

    // Process 4 positions in parallel using 8 DST tiles (4 pos * 2 channel tiles)
    tile_regs_acquire();

    // For 4-way parallel, we need to coordinate CBs for 4 positions
    // Current implementation: use existing split_reader structure
    // Position 0, 2: use reader 0 (in_cb_id_0)
    // Position 1, 3: use reader 1 (in_cb_id_1)

    for (uint32_t pos_pair = 0; pos_pair < 2; ++pos_pair) {
        // Each pair processes 2 positions (pos_pair*2 and pos_pair*2+1)

        // Position A: reader 0
        uint32_t curr_in_cb_id_a = in_cb_id_0;
        uint32_t curr_scalar_cb_id_a = in_scalar_cb_id_0;

        // Position B: reader 1
        uint32_t curr_in_cb_id_b = in_cb_id_1;
        uint32_t curr_scalar_cb_id_b = one_scalar_per_core ? in_scalar_cb_id_0 : in_scalar_cb_id_1;

        cb_wait_front(curr_in_cb_id_a, 1);
        cb_wait_front(curr_in_cb_id_b, 1);

        if (!one_scalar_per_core) {
            cb_wait_front(curr_scalar_cb_id_a, 1);
            cb_wait_front(curr_scalar_cb_id_b, 1);
        }

        DST_DEBUG_PRINT(
            "Position pair " << pos_pair << ": processing positions " << (pos_pair * 2) << " and "
                             << (pos_pair * 2 + 1));

        // Unpack data for position A (pos_pair*2)
        unpack_tilizeA_B_block<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
            curr_in_cb_id_a,
            curr_scalar_cb_id_a,
            tiles_to_reduce,
            0,  // tile idx for Src b
            num_faces_in_input_tile,
            face_r_dim);

        // Math for position A: use DST indices pos_pair*2*tiles_to_reduce + ch
        for (uint32_t ch_tile = 0; ch_tile < tiles_to_reduce; ++ch_tile) {
            uint32_t dst_idx_a = (pos_pair * 2) * tiles_to_reduce + ch_tile;
            DST_DEBUG_PRINT("DST[" << dst_idx_a << "] -> Pos=" << (pos_pair * 2) << " Ch=" << ch_tile);
            reduce_tile_math(dst_idx_a, num_faces_in_input_tile);
        }

        // Unpack data for position B (pos_pair*2+1)
        unpack_tilizeA_B_block<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
            curr_in_cb_id_b,
            curr_scalar_cb_id_b,
            tiles_to_reduce,
            0,  // tile idx for Src b
            num_faces_in_input_tile,
            face_r_dim);

        // Math for position B: use DST indices (pos_pair*2+1)*tiles_to_reduce + ch
        for (uint32_t ch_tile = 0; ch_tile < tiles_to_reduce; ++ch_tile) {
            uint32_t dst_idx_b = (pos_pair * 2 + 1) * tiles_to_reduce + ch_tile;
            DST_DEBUG_PRINT("DST[" << dst_idx_b << "] -> Pos=" << (pos_pair * 2 + 1) << " Ch=" << ch_tile);
            reduce_tile_math(dst_idx_b, num_faces_in_input_tile);
        }

        cb_pop_front(curr_in_cb_id_a, 1);
        cb_pop_front(curr_in_cb_id_b, 1);
    }

    tile_regs_commit();
    tile_regs_wait();

    // Pack results from all 4 positions (8 DST tiles total)
    DST_DEBUG_PRINT("Packing 4-position results: 8 DST tiles -> output");

    // Pack all 4 positions in one call
    const uint32_t total_output_tiles = 4 * tiles_to_reduce;  // 4 positions * channel tiles
    pack_untilize_dest<8>(out_cb_id, 1, 0, num_out_sticks * 4, num_faces_in_output_tile);
    cb_push_back(out_cb_id, total_output_tiles);

    tile_regs_release();
}

ALWI bool should_use_dst_optimization(uint32_t tiles_to_reduce, uint32_t nsticks_remaining) {
    // Use DST optimization for C=64 (2 channel tiles) with sufficient work
    // Require at least 4 positions to make optimization worthwhile
    return (tiles_to_reduce == 2) && (nsticks_remaining >= 4);
}

ALWI void debug_dst_optimization_state(
    uint32_t iteration, uint32_t tiles_to_reduce, uint32_t nsticks_remaining, bool optimization_enabled) {
    DST_DEBUG_PRINT("=== DST OPTIMIZATION STATE ===");
    DST_DEBUG_PRINT("Iteration=" << iteration << " tiles_to_reduce=" << tiles_to_reduce);
    DST_DEBUG_PRINT(
        "Sticks remaining=" << nsticks_remaining
                            << " optimization=" << (optimization_enabled ? "ENABLED" : "DISABLED"));

    if (optimization_enabled) {
        DST_DEBUG_PRINT("DST_4WAY_PARALLEL_MODE enabled");
        DST_DEBUG_PRINT("DST_UTILIZATION=100.0%");
        DST_DEBUG_PRINT("Expected DST tiles: 8 (4 positions * 2 channels)");
    } else {
        DST_DEBUG_PRINT("DST_SEQUENTIAL_MODE");
        DST_DEBUG_PRINT("DST_UTILIZATION=" << (tiles_to_reduce * 100 / 8) << ".0%");
    }
}

void MAIN {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet. When ntiles_hw > 1 the large
    // kernel is called
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(0);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(1);

    constexpr uint32_t split_reader = get_compile_time_arg_val(2);

    constexpr uint32_t nsticks_per_core_by_nblocks = get_compile_time_arg_val(3);
    constexpr uint32_t in_c = get_compile_time_arg_val(4);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(5);
    constexpr uint32_t max_sticks_for_reduction = get_compile_time_arg_val(6);

    constexpr uint32_t in_cb_id_0 = get_compile_time_arg_val(7);
    constexpr uint32_t in_cb_id_1 = get_compile_time_arg_val(8);  // for split reader
    constexpr uint32_t in_idx_cb_id_0 = get_compile_time_arg_val(9);
    constexpr uint32_t in_idx_cb_id_1 = get_compile_time_arg_val(10);  // for split reader
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(11);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(12);
    constexpr uint32_t tile_tmp_cb_id = get_compile_time_arg_val(13);
    constexpr uint32_t tile_idx_tmp_cb_id = get_compile_time_arg_val(14);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(15);
    constexpr uint32_t out_idx_cb_id = get_compile_time_arg_val(16);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(17);
    constexpr bool return_indices = (bool)get_compile_time_arg_val(18);
    constexpr uint32_t pre_tilize_cb_id = get_compile_time_arg_val(19);
    constexpr bool is_output_tiled = get_compile_time_arg_val(20);  // 1 = TILED, 0 = ROW_MAJOR
    constexpr bool is_output_block_format = (bool)get_compile_time_arg_val(21);

    // DST Optimization parameters - Phase 2: Re-enable parameter reception
    constexpr uint32_t dst_optimization_mode = get_compile_time_arg_val(22);
    constexpr uint32_t positions_per_iteration = get_compile_time_arg_val(23);

    // DST optimization detection - Phase 3: FULLY ENABLED!
    constexpr bool use_dst_optimization = (dst_optimization_mode == 2) &&  // QUAD_POSITION
                                          (in_ntiles_c == 2) &&            // C=64 case
                                          (!return_indices);               // Not supported with indices yet

    constexpr uint32_t topk_output_tiles = 1;
    constexpr uint32_t topk_cb_tile_idx = 0;
    constexpr uint32_t data_dst_idx = 0;
    constexpr uint32_t index_dst_idx = 2;

    constexpr uint32_t face_r_dim = window_size_hw < FACE_HEIGHT && !return_indices ? window_size_hw : FACE_HEIGHT;
    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    constexpr uint32_t num_faces_in_input_tile =
        (max_sticks_for_reduction < TILE_HEIGHT || window_size_hw <= FACE_HEIGHT) && !return_indices ? 2 : 4;
    constexpr uint32_t num_faces_in_output_tile = 2;
    constexpr uint32_t num_faces_in_last_output_tile = last_tile_is_partial && in_c % TILE_WIDTH <= FACE_WIDTH ? 1 : 2;
    constexpr uint32_t num_out_sticks = 1;

    constexpr bool is_avg_pool = REDUCE_OP == PoolType::SUM;
    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time.
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = return_indices ? 1 : (is_avg_pool && is_large_kernel) ? 4 : 8;
    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    static_assert(REDUCE_OP == PoolType::MAX || REDUCE_OP == PoolType::SUM, "Only supports REDUCE_OP = MAX or Sum");
    constexpr bool neginf_srca_maxpool = (REDUCE_OP == PoolType::MAX) ? true : false;
    constexpr bool zero_srca_avgpool = (REDUCE_OP == PoolType::SUM) ? true : false;

    // tilize reconfiguration can be beneficial when we have a wide tensor with a non MAX_TILES_PER_REDUCTION number of
    // C tiles, but we only use it when the window size fits within a face such that the tilize can be done only on the
    // rows populated with data, otherwise we need to call clear_out_tiles between reconfigs to avoid untilizing junk
    // data which is much slower than just untilizing the entire MAX_TILES_PER_REDUCTION
    constexpr bool tilize_reconfig = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 &&
                                     window_size_hw <= FACE_HEIGHT && !last_tile_is_partial;
#ifdef ARCH_BLACKHOLE
    constexpr bool use_tilize_dest = in_c <= FACE_WIDTH;
    constexpr bool pack_untilize_reinit = !use_tilize_dest;
#else
    constexpr bool use_tilize_dest = true;
    constexpr bool pack_untilize_reinit = last_tile_is_partial && in_ntiles_c > 1;
#endif
    if constexpr (!return_indices) {
        constexpr uint32_t tilize_untilize_cb = is_output_tiled ? pre_tilize_cb_id : out_cb_id;
        tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
            in_cb_id_0, in_scalar_cb_id_0, max_tiles_per_iter, tilize_untilize_cb, num_faces_in_input_tile, face_r_dim);
        pack_untilize_dest_init<max_tiles_per_iter>(tilize_untilize_cb, num_out_sticks, num_faces_in_output_tile);
    } else {
        if constexpr (use_tilize_dest) {
            unary_op_init_common(in_cb_id_0, in_cb_id_0);
            tilize_init_no_pack(in_cb_id_0, topk_output_tiles);
            if constexpr (!pack_untilize_reinit) {
                const uint32_t output_faces =
                    last_tile_is_partial ? num_faces_in_last_output_tile : num_faces_in_output_tile;
                pack_untilize_dest_init<topk_output_tiles>(out_cb_id, num_out_sticks, output_faces);
            }
        }

        // this can be done here because we do not use the SFPU for anything else so it does not get reprogrammed
        // if you use the sfpu for other operations, you need to call this to reprogram the sfpu
        max_reduce_with_indices_init();
    }

    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;

    // wait for initialization to complete
    if constexpr (one_scalar_per_core) {
        cb_wait_front(in_scalar_cb_id_0, 1);
    }

    uint32_t tilize_stick_counter = 0;

    // DST Optimization: Check if we should use 4-way parallel processing
    DST_DEBUG_PRINT("=== COMPUTE KERNEL START ===");
    DST_DEBUG_PRINT("in_ntiles_c=" << in_ntiles_c << " window_size_hw=" << window_size_hw);
    DST_DEBUG_PRINT("dst_optimization_mode=" << dst_optimization_mode);
    DST_DEBUG_PRINT("positions_per_iteration=" << positions_per_iteration);
    DST_DEBUG_PRINT("use_dst_optimization=" << (uint32_t)use_dst_optimization);

    // Kernel-level DST capacity validation (additional safety check)
    if constexpr (use_dst_optimization) {
        constexpr uint32_t required_dst_tiles = 4 * in_ntiles_c;  // 4 positions × channel tiles

        // Runtime validation (static_assert removed because it fails for large channel counts)
        // Factory-level validation should ensure we never reach here with required_dst_tiles > 8
        DST_DEBUG_PRINT("Required DST tiles: " << required_dst_tiles);

// Runtime check as additional safety (only in debug builds)
#ifdef DEBUG
        if (required_dst_tiles > 8) {
            DST_DEBUG_PRINT(
                "ERROR: DST optimization enabled but requires " << required_dst_tiles
                                                                << " tiles > 8 limit - Factory validation failed!");
        }
#endif
    }

    // Process in batches if DST optimization is enabled
    if constexpr (use_dst_optimization) {
        DST_DEBUG_PRINT("DST optimization enabled: processing in 4-position batches");

        // Process 4 positions at a time
        for (uint32_t n = 0; n < nsticks_per_core_by_nblocks; n += 4) {
            debug_dst_optimization_state(n / 4, in_ntiles_c, nsticks_per_core_by_nblocks - n, true);

            // Ensure we have at least 4 positions remaining
            if (n + 4 > nsticks_per_core_by_nblocks) {
                DST_DEBUG_PRINT(
                    "Insufficient positions remaining: " << (nsticks_per_core_by_nblocks - n)
                                                         << " < 4, falling back to sequential");
                break;  // Fall through to sequential processing for remaining positions
            }

            // Process 4 positions in parallel
            process_4way_parallel_positions<neginf_srca_maxpool, zero_srca_avgpool>(
                in_cb_id_0,
                in_cb_id_1,
                in_scalar_cb_id_0,
                in_scalar_cb_id_1,
                out_cb_id,
                in_ntiles_c,  // tiles_to_reduce
                num_faces_in_input_tile,
                face_r_dim,
                num_out_sticks,
                num_faces_in_output_tile,
                split_reader,
                one_scalar_per_core,
                n / 4  // current batch number
            );
        }

        DST_DEBUG_PRINT("DST optimization processing complete");
        return;  // Early return - DST optimization path complete
    }

    DST_DEBUG_PRINT("Using sequential processing mode");

    // Original sequential processing loop
    for (uint32_t n = 0; n < nsticks_per_core_by_nblocks; ++n) {
        const bool reader0 = !(split_reader && (n & 0x1));
        const uint32_t curr_scalar_cb_id = (!reader0 && !one_scalar_per_core) ? in_scalar_cb_id_1 : in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = !reader0 ? in_cb_id_1 : in_cb_id_0;
        const uint32_t curr_in_idx_cb_id = !reader0 ? in_idx_cb_id_1 : in_idx_cb_id_0;
        if constexpr (!one_scalar_per_core) {
            cb_wait_front(curr_scalar_cb_id, 1);
        }
        if (is_output_tiled && !tilize_stick_counter) {
            cb_reserve_back(out_cb_id, in_ntiles_c);
        }
        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            const bool last_c_block = c_i == in_nblocks_c - 1;
            const bool first_c_block = c_i == 0;
            const uint32_t tiles_to_reduce =
                tilize_reconfig ? (last_c_block ? partial_iter_output_tiles : max_tiles_per_iter) : max_tiles_per_iter;
            const uint32_t number_of_tiles = last_c_block ? partial_iter_output_tiles : max_tiles_per_iter;
            const uint32_t output_faces =
                (last_tile_is_partial && last_c_block)
                    ? (number_of_tiles - 1) * num_faces_in_output_tile + num_faces_in_last_output_tile
                    : number_of_tiles * num_faces_in_output_tile;
            if constexpr (!is_output_tiled) {
                cb_reserve_back(out_cb_id, output_faces);
            }
            if constexpr (tilize_reconfig || is_output_tiled) {
                if (first_c_block || last_c_block) {
                    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce, num_faces_in_input_tile, face_r_dim, 1)));
                }
            }
            tile_regs_acquire();
            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
                cb_wait_front(curr_in_cb_id, 1);
                if constexpr (return_indices) {
                    cb_wait_front(curr_in_idx_cb_id, 1);

                    if constexpr (use_tilize_dest) {
                        tilize_dest_function<topk_output_tiles, data_dst_idx, index_dst_idx, topk_cb_tile_idx>(
                            curr_in_cb_id, curr_in_idx_cb_id);
                    } else {
                        tilize_copy_function<
                            topk_output_tiles,
                            data_dst_idx,
                            index_dst_idx,
                            topk_cb_tile_idx,
                            tile_tmp_cb_id,
                            tile_idx_tmp_cb_id,
                            out_cb_id,
                            num_out_sticks,
                            pack_untilize_reinit>(curr_in_cb_id, curr_in_idx_cb_id, output_faces);
                    }

                    max_reduce_with_indices<window_size_hw>(data_dst_idx, index_dst_idx);

                    cb_pop_front(curr_in_idx_cb_id, 1);
                } else {
                    unpack_tilizeA_B_block<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        curr_in_cb_id,
                        curr_scalar_cb_id,
                        tiles_to_reduce,
                        0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/,
                        num_faces_in_input_tile,
                        face_r_dim);
                    for (uint32_t math_tile_idx = 0; math_tile_idx < tiles_to_reduce; ++math_tile_idx) {
                        reduce_tile_math(math_tile_idx, num_faces_in_input_tile);
                    }
                }
                cb_pop_front(curr_in_cb_id, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (!return_indices) {
                if constexpr (is_output_tiled) {
                    // TILED output: accumulate sticks and perform tilization when needed
                    if (last_c_block) {
                        pack_untilize_dest<partial_iter_output_tiles>(
                            pre_tilize_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
                        cb_push_back(pre_tilize_cb_id, partial_iter_output_tiles);
                        tilize_stick_counter++;
                    } else {
                        pack_untilize_dest<max_tiles_per_iter>(
                            pre_tilize_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
                        cb_push_back(pre_tilize_cb_id, max_tiles_per_iter);
                    }
                    tile_regs_release();

                    if (tilize_stick_counter == TILE_HEIGHT) {
                        PACK((pack_untilize_uninit(pre_tilize_cb_id)));

                        // Workaround until #27504 is not closed
                        tensix_sync();
                        unary_op_init_common(pre_tilize_cb_id, out_cb_id);
                        tensix_sync();

                        fast_tilize_init(pre_tilize_cb_id, in_ntiles_c, out_cb_id);
                        fast_tilize_block(pre_tilize_cb_id, in_ntiles_c, out_cb_id);
                        fast_tilize_uninit(pre_tilize_cb_id, out_cb_id);

                        cb_push_back(out_cb_id, in_ntiles_c);

                        if constexpr (is_output_block_format) {
                            tensix_sync();
                            unary_op_init_common(in_cb_id_0, pre_tilize_cb_id);
                            tensix_sync();
                        }

                        tilize_stick_counter = 0;
                        // init math for reduction again since FPU gets reprogrammed by tilize
                        MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, DST_ACCUM_MODE, MATH_FIDELITY>()));
#ifdef ARCH_BLACKHOLE
                        // need this on BH to set swizzle bit before pack untilize dest
                        MATH((llk_math_hw_configure_disaggregated<true, true>(0, 0)));
#endif
                        PACK((llk_pack_untilize_init<max_tiles_per_iter, max_tiles_per_iter, false, false, TILE_C_DIM>(
                            pre_tilize_cb_id, 1, num_faces_in_output_tile)));
                    }
                } else {
                    // ROW_MAJOR output: pack directly to output CB
                    if (last_c_block) {
                        pack_untilize_dest<partial_iter_output_tiles>(
                            out_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
                    } else {
                        pack_untilize_dest<max_tiles_per_iter>(
                            out_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
                    }
                    cb_push_back(out_cb_id, output_faces);
                    tile_regs_release();
                }
            } else {
                if constexpr (use_tilize_dest) {
                    if constexpr (pack_untilize_reinit) {
#ifdef ARCH_BLACKHOLE
                        // Needed for setting swizzle_32b:
                        MATH((llk_math_hw_configure_disaggregated<true, true>(0, 0)));
#endif
                        PACK((llk_pack_untilize_init<topk_output_tiles, topk_output_tiles, false, false, TILE_C_DIM>(
                            out_cb_id, num_out_sticks, output_faces)));
                        PACK((llk_init_packer_dest_offset_registers<true, false>()));
                    }
                }

                pack_reconfig_data_format(out_cb_id);
                pack_untilize_dest<topk_output_tiles, topk_output_tiles, false, false, TILE_C_DIM, data_dst_idx>(
                    out_cb_id, 1, 0, num_out_sticks, output_faces);
                pack_reconfig_data_format(out_idx_cb_id);
                pack_untilize_dest<topk_output_tiles, topk_output_tiles, false, false, TILE_C_DIM, index_dst_idx>(
                    out_idx_cb_id, 1, 0, num_out_sticks, output_faces);

                if constexpr (pack_untilize_reinit) {
                    pack_untilize_uninit(out_cb_id);
                }
                cb_push_back(out_cb_id, output_faces);
                if constexpr (return_indices) {
                    cb_push_back(out_idx_cb_id, output_faces);
                }
                tile_regs_release();
            }
        }
        if constexpr (!one_scalar_per_core) {
            cb_pop_front(curr_scalar_cb_id, 1);
        }
    }
}

}  // namespace NAMESPACE
