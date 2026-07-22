// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/layernorm.h"
#include "api/compute/transpose.h"
#include "api/compute/welford.h"
#include "api/compute/eltwise_binary.h"
#include "ttnn/operations/normalization/kernel_util/compute/combine_welford.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "api/dataflow/circular_buffer.h"

/**
 * @brief This kernel computes layernorm for sharded tensors using
 *        Welford's algorithm for the mean and variance calculations
 *
 * @details Computes layernorm(x) = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
 *
 * There are two flavors of sharded layernorm. The details
 * here are for row-major tensors, but the logic for column-major
 * is the same (core/tensor rows are replaced by core/tensor columns):
 * 1. Single-stage reduce:
 *   - Each core gets a width slice (of size `block_wt` tiles) of
 *     one or more rows of the tensor (the number of rows each core
 *     is assigned is `num_tiles_per_allgather_worker`)
 *   - Each core computes its partial mean and variance of its slices
 *     for all rows it is assigned (using Welford's algorithm) and pushes
 *     the interleaved mean and variance results to `cb_ex_partial_id`.
 *     This produces 1 mean tile and 1 variance tile per tile row
 *   - The reader kernels populate `cb_ex_external_id` with the core's
 *     partial result + the other partial results from cores in the
 *     same core row for each row the core is assigned
 *   - Each core combines all partial results for its row(s) in
 *     `cb_ex_external_id` into `cb_ex_id`. `cb_ex_id` contains 1 mean tile
 *     followed by 1 tile of 1/sqrt(var + eps) for each assigned row
 *   - The core row's sender core (the first column of cores) collects
 *     all of these combined tiles and multicasts to all cores in the row
 *     into `cb_ex_global_id`
 * 2. Two-stage reduce:
 *   - Used for width-sharded tensors, where each core has a
 *     tensor-height-tall slice of the tensor
 *   - The top row of cores are designated as "second-stage readers"
 *   - As in single-stage reduce, each core in a row computes
 *     its partial mean and variance for each of its assigned rows
 *     for its width shard then combines it with the other partials
 *     in its core row. This is the first stage of the reduce.
 *   - Second stage of the reduce: The reader kernels add the
 *     first stage's combined results into the second stage readers'
 *     `cb_ex_external_id`. The results in `cb_ex_external_id` are combined
 *     in the same way.
 *   - The final combined results are collected by the sender core
 *     and multicasted to all cores into `cb_ex_global_id`
 *
 * After one of the two reduce paths above, the rest of the layernorm
 * calculation is done using the global mean and 1/sqrt(var + eps) results.
 *
 * @note Depending on the tensor and core grid shape, some cores may
 *       not participate in the combine (i.e., `is_allgather_worker`
 *       will be false). These cores do their partial reduction and
 *       receive the multicasted global results and perform
 *       the rest of layernorm for their row(s) width slices.
 */
namespace {
// Element count (Welford weight) of the width block at a given GLOBAL width-block index: a full block
// before the logical boundary, the partially-valid boundary block itself, or zero for a pure-padding
// block past the logical width. The boundary is the single global position where the logical width ends.
inline uint32_t block_set_size(
    const uint32_t global_block_index,
    const uint32_t boundary_width_index,
    const uint32_t block_w,
    const uint32_t last_block_w) {
    if (global_block_index < boundary_width_index) {
        return block_w;
    }
    if (global_block_index == boundary_width_index) {
        return last_block_w;
    }
    return 0;
}

// Total logical width of a first-stage row (num_blocks_first_stage consecutive blocks starting at
// row * num_blocks_first_stage): a full row before the boundary row, zero for a row entirely past it,
// or the summed real widths of the boundary row (full blocks up to the boundary plus the partial block).
inline uint32_t row_set_size(
    const uint32_t row,
    const uint32_t num_blocks_first_stage,
    const uint32_t boundary_width_index,
    const uint32_t block_w,
    const uint32_t last_block_w) {
    const uint32_t boundary_row = boundary_width_index / num_blocks_first_stage;
    if (row < boundary_row) {
        return num_blocks_first_stage * block_w;
    }
    if (row > boundary_row) {
        return 0;
    }
    const uint32_t full_blocks_before_boundary = boundary_width_index - row * num_blocks_first_stage;
    return full_blocks_before_boundary * block_w + last_block_w;
}

// Weight of the b-th block in this core's Welford combine, by its true logical width.
inline uint32_t get_next_set_size(
    const uint32_t block,
    const bool is_second_stage_reader,
    const uint32_t num_blocks_first_stage,
    const uint32_t own_row,
    const uint32_t boundary_width_index,
    const uint32_t block_w,
    const uint32_t last_block_w) {
    if (is_second_stage_reader) {
        // The first num_blocks_first_stage blocks are this reader's own row, streamed in width order;
        // the rest are the per-row combined results of the other rows, streamed in row order (for row
        // major, from this reader's core column). Weight own-row blocks by their global block width and
        // each other-row result by that row's total logical width.
        if (block < num_blocks_first_stage) {
            return block_set_size(
                own_row * num_blocks_first_stage + block, boundary_width_index, block_w, last_block_w);
        }
        const uint32_t row = own_row + (block - num_blocks_first_stage) + 1;
        return row_set_size(row, num_blocks_first_stage, boundary_width_index, block_w, last_block_w);
    }

    // First-stage worker (or single-stage): the blocks are this core's own row, in width order.
    return block_set_size(own_row * num_blocks_first_stage + block, boundary_width_index, block_w, last_block_w);
}
}  // namespace
void kernel_main() {
    // ============================================================================
    // Kernel setup
    // ============================================================================

    // ---------------------------------------------------------------------------
    // Compile-time arguments
    // ---------------------------------------------------------------------------
    constexpr uint32_t is_top_row = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(3);
    constexpr uint32_t block_wt = get_compile_time_arg_val(5);
    constexpr uint32_t block_ht_const = get_compile_time_arg_val(4);
    volatile uint32_t block_ht_volatile = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_wt_const = get_compile_time_arg_val(6);
    volatile uint32_t subblock_wt_volatile = get_compile_time_arg_val(6);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(7);
    const bool is_allgather_worker = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(9);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(10) == 1;
    constexpr bool FLOAT32_REDUCTION = get_compile_time_arg_val(11) == 1;
    constexpr bool LEGACY_RSQRT = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(13);
    constexpr uint32_t tile_width = get_compile_time_arg_val(14);
    constexpr uint32_t last_tile_w = get_compile_time_arg_val(15);
    constexpr uint32_t W = get_compile_time_arg_val(16);
    constexpr uint32_t eps = get_compile_time_arg_val(17);
    constexpr uint32_t per_core_recip_lut_size = get_compile_time_arg_val(18);
    // Valid (logical) tile count of the final width block: the number of its tiles that hold any
    // logical data, the last of which may be only partially valid. Fewer than block_wt when the
    // logical width does not fill the width blocks evenly (each block spans a whole number of tiles).
    // A partial boundary tile is counted as a valid tile here; its valid-column count is carried
    // separately in last_tile_w and combined into last_block_w.
    // For example, w=96 gives 3 tiles, which sharded on two cores leaves two real tiles on the first
    // core and one real tile plus one padding tile on the second. For w=80 (also 3 tiles), the second
    // core owns last_block_wt = 1 tile that is itself partial (last_tile_w = 16 valid columns) plus
    // one padding tile.
    constexpr uint32_t last_block_wt = get_compile_time_arg_val(19);

    // ---------------------------------------------------------------------------
    // CB definitions
    // ---------------------------------------------------------------------------
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t cb_x = tt::CBIndex::c_24;          // x minus mean
    constexpr uint32_t cb_xmm_id = tt::CBIndex::c_18;     // x minus mean
    constexpr uint32_t cb_ex_partial_id = tt::CBIndex::c_8;  // Interleaved E[x] and Var[x] partial results
    constexpr uint32_t cb_ex_id = tt::CBIndex::c_9;          // Interleaved E[x] and Var[x] global reduce
    constexpr uint32_t cb_ex_external_id = tt::CBIndex::c_10;
    constexpr uint32_t cb_ex_global_id = tt::CBIndex::c_15;  // Interleaved E[x] and Var[x] final global mcast result
    constexpr uint32_t cb_transpose_id = tt::CBIndex::c_22;  // Transpose interleaved E[x] and Var[x] to columns
                                                             // (workaround for bug in transpose_dest)
    constexpr uint32_t cb_fusion_id = tt::CBIndex::c_18;     // stream gamma/beta
    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;
    constexpr uint32_t cb_reciprocals = tt::CBIndex::c_25;  // LUT of pre-computed reciprocals for Welford's algorithm

    CircularBuffer cb_gamma(cb_gamma_id);
    CircularBuffer cb_beta(cb_beta_id);
    CircularBuffer cb_xmm(cb_xmm_id);
    CircularBuffer cb_ex_partial(cb_ex_partial_id);
    CircularBuffer cb_ex(cb_ex_id);
    CircularBuffer cb_ex_external(cb_ex_external_id);
    CircularBuffer cb_ex_global(cb_ex_global_id);
    CircularBuffer cb_transpose(cb_transpose_id);
    CircularBuffer cb_fusion(cb_fusion_id);
    CircularBuffer cb_out(cb_out_id);

    constexpr uint32_t cb_im_id = (do_gamma | do_beta) ? cb_x : cb_out_id;
    CircularBuffer cb_im(cb_im_id);
    constexpr uint32_t cb_outgamma_id = do_beta ? cb_fusion_id : cb_out_id;
    CircularBuffer cb_outgamma(cb_outgamma_id);
#ifdef FUSE_PRE_ADD
    constexpr uint32_t cb_in_id = cb_x;
#else
    constexpr uint32_t cb_in_id = cb_in0;
#endif
    CircularBuffer cb_in(cb_in_id);

    // Welford-fp32 alias of cb_in_id. When welford_fp32_alias is true, cb_x_welford_named points
    // to c_29, a separate buffer index sharing cb_in_id's SRAM but configured with UnpackToDestFp32,
    // so Welford's transpose_tile preserves fp32 precision in DEST. The two aliased indices
    // have independent read/write pointers so the fused path pushes both side by side; the non-fused
    // path reads c_0 (sharded) without read/write pointer manipulation, and so does the alias.
    constexpr bool welford_fp32_alias = get_named_compile_time_arg_val("welford_fp32_alias") != 0;
    constexpr auto cb_x_welford_named = get_named_compile_time_arg_val("cb_x_welford");
    constexpr auto cb_x_welford_id = welford_fp32_alias ? cb_x_welford_named : cb_in_id;
    CircularBuffer cb_x_welford(cb_x_welford_id);

    // ---------------------------------------------------------------------------
    // Derived quantities
    // ---------------------------------------------------------------------------
    // set block_ht to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_ht = (block_wt == 1) ? block_ht_volatile : block_ht_const;
    const uint32_t subblock_wt = (block_wt <= 2) ? subblock_wt_volatile : subblock_wt_const;

    // This core's real (logical) column count. Welford has no per-column mask, so each core must reduce
    // over exactly its logical columns: cores before the last own a full block_w, and the last real
    // core owns the remaining logical columns; a whole number of full tiles plus, when the logical
    // width is not tile-aligned, a final partial tile. The reduce must stop there rather than at the
    // physical shard end, which carries padding tiles. This is the only per-core quantity that differs
    // for the partial final shard.
    // On all-to-all workers indices 1 through 3 hold the two-stage-reduce args, so this value is appended
    // at index 4, whereas on other workers it immediately follows the reduce-tile count at index 1.
    const uint32_t welford_reduce_w = get_arg_val<uint32_t>(is_allgather_worker ? 4 : 1);
    const uint32_t partial_reduce_W = welford_reduce_w;

    // Split the Welford reduction into full tiles and a final partial tile (present only when the
    // logical width is not a multiple of the tile width).
    const uint32_t num_full_welford_tiles = welford_reduce_w / tile_width;
    const uint32_t partial_welford_tile_w = welford_reduce_w % tile_width;

    // This is the number of tile rows to process
    const uint32_t num_tiles_per_allgather_worker = is_allgather_worker ? get_arg_val<uint32_t>(1) : 0;

    // These are for two-stage reductions
    const bool use_two_stage_reduce = is_allgather_worker ? get_arg_val<uint32_t>(2) == 1 : false;
    const bool is_second_stage_reader = is_allgather_worker ? get_arg_val<uint32_t>(3) == 1 : false;
    constexpr uint32_t block_w = block_wt * tile_width;
    // Width (valid columns) of the final width block, weighting it in the cross-core combine. The
    // final block owns last_block_wt tiles (<= block_wt), the last of which has last_tile_w valid
    // columns; the other blocks each own a full block_w.
    constexpr uint32_t last_block_w = (last_block_wt - 1) * tile_width + last_tile_w;
    // Global width-block index of the partial boundary block and this core's own width-block index,
    // read only on all-to-all workers (the cores that run the cross-core combine). own_row is this
    // core's first-stage row; a width shard's global index is own_row * num_blocks_first_stage + its
    // position within the row. These let the combine weight each block/row by its true logical width.
    const uint32_t boundary_width_index = is_allgather_worker ? get_arg_val<uint32_t>(5) : 0;
    const uint32_t my_width_index = is_allgather_worker ? get_arg_val<uint32_t>(6) : 0;
    const uint32_t own_row = my_width_index / num_blocks_first_stage;

    // The number of blocks to combine.
    // If we're the second stage reader, we're reducing the
    // entire tensor width.
    // If we're part of a two-stage reduce and not a reader,
    // or we're part of a single-stage reduce, we're reducing
    // width is only along our row
    uint32_t num_blocks_combine =
        is_second_stage_reader ? num_blocks_first_stage + num_blocks_second_stage - 1 : num_blocks_first_stage;

    // Number of tiles for block_ht results (interleaved mean and var)
    const uint32_t num_block_ht_result_tiles = 2 * block_ht;

    // Only used for the transpose workaround
    constexpr uint32_t num_dest_regs = FLOAT32_DTYPE ? 4 : 8;

    // Welford destination registers
    constexpr uint32_t welford_input_dst = 0;
    constexpr uint32_t welford_mean_dst = 1;
    constexpr uint32_t welford_var_dst = 2;

    // Pointer to the reciprocal LUT

    using recip_lut_t = std::array<uint32_t, per_core_recip_lut_size>;
    auto p_reciprocals = norm::kernel_util::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0);

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

    // ============================================================================
    // Main kernel logic
    // ============================================================================

    // ---------------------------------------------------------------------------
    // Op initialization
    // ---------------------------------------------------------------------------
#ifdef FUSE_PRE_ADD
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_in_id);
#else
    unary_op_init_common(cb_in_id, cb_ex_partial_id);
#endif

    // ---------------------------------------------------------------------------
    // Pre-add x + y
    // ---------------------------------------------------------------------------
#ifdef FUSE_PRE_ADD
    reconfig_data_format_srcb(cb_in0, cb_in1);
    add_init(cb_in0, cb_in1);
    cb_in.reserve_back(num_tiles_per_block);
    if constexpr (welford_fp32_alias) {
        // Must be done in the compute kernel: on the fused path compute is the producer of cb_in_id
        // via the add_tiles -> pack_tile sequence below; the reader never writes cb_in_id.
        // cb_x_welford_id shares cb_in_id's SRAM but has its own read/write pointers, so reserve and push
        // both indices side by side. pack_tile writes once via cb_in_id's wr_ptr; the alias lets the
        // welford section wait_front on c_29 independently of cb_in_id.
        cb_x_welford.reserve_back(num_tiles_per_block);
    }
    for (uint32_t i = 0; i < block_ht; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                add_tiles(cb_in0, cb_in1, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                pack_tile(sbi, cb_in_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_wt;
        }
        index_h_offset += block_wt;
    }
    cb_in.push_back(num_tiles_per_block);
    cb_in.wait_front(num_tiles_per_block);
    if constexpr (welford_fp32_alias) {
        cb_x_welford.push_back(num_tiles_per_block);
        cb_x_welford.wait_front(num_tiles_per_block);
    }
#endif

    // ---------------------------------------------------------------------------
    // Compute E[x] and Var[x] using Welford's algorithm
    // ---------------------------------------------------------------------------
    reconfig_data_format_srca(cb_x_welford_id);
    cb_ex_partial.reserve_back(num_block_ht_result_tiles);
    // Reconfigure the transpose op for the welford intake CB. When the alias is active,
    // cb_x_welford_id has UnpackToDestFp32 mode so transpose_tile preserves fp32 precision.
    transpose_init(cb_x_welford_id);
    welford_init();
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_ht; i++) {
        tile_regs_acquire();
        welford_clear();
        uint32_t sample_idx = 0;

        // Do the full Welford tiles
        for (uint32_t w = 0; w < num_full_welford_tiles; w++) {
            if constexpr (welford_fp32_alias) {
                // SFPU replay slots [0, 32) currently hold the welford recurrence (welford uses
                // the full 32-slot math-thread replay buffer; the recovery block below re-records
                // all of it after each transpose). transpose_init re-records slots
                // [16, 32) with the transpose-dest setup so transpose_tile below can replay them.
                transpose_init(cb_x_welford_id);
            }
            transpose_tile(cb_x_welford_id, w + index_h_offset, welford_input_dst);
            if constexpr (welford_fp32_alias) {
                // transpose_tile took the UnpackToDestFp32 path. Its math-side init clobbered
                // the welford recurrence at SFPU replay slots [16, 32).
                // welford_init<WelfordInitMode::PreserveStats>() re-records all 32 slots with
                // the welford recurrence; PreserveStats keeps the running mean / M2 accumulator
                // in LREG4/5. UNPACK A is left in transpose=1;
                // welford_update is pure SFPU and does not consume that state, and the next
                // iteration's transpose_init reprograms it.
                welford_init<WelfordInitMode::PreserveStats>();
            }
            welford_update<per_core_recip_lut_size>(welford_input_dst, sample_idx, *p_reciprocals);
            sample_idx += tile_width;
        }
        // Do the partial Welford tile, if any. It is the tile immediately after this core's full tiles
        // (index_h_offset + num_full_welford_tiles), i.e. the last real tile of this core's logical
        // columns; not necessarily the last physical tile of the shard (block_wt - 1), which on the
        // final core is a pure-padding tile when the width is split across cores.
        if (partial_welford_tile_w > 0) {
            if constexpr (welford_fp32_alias) {
                transpose_init(cb_x_welford_id);
            }
            transpose_tile(cb_x_welford_id, index_h_offset + num_full_welford_tiles, welford_input_dst);
            if constexpr (welford_fp32_alias) {
                welford_init<WelfordInitMode::PreserveStats>();
            }
            welford_update_rows<per_core_recip_lut_size>(
                welford_input_dst, sample_idx, 0, partial_welford_tile_w, *p_reciprocals);
        }
        welford_finalize_to_row<per_core_recip_lut_size>(welford_mean_dst, partial_reduce_W - 1, *p_reciprocals);
        // We should transpose back to columns here
        // However, transpose_dest() is currently buggy.
        // So we transpose to an intermediate CB downstream
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(welford_mean_dst, cb_ex_partial_id);
        pack_tile(welford_var_dst, cb_ex_partial_id);
        tile_regs_release();
        index_h_offset += block_wt;
    }
    cb_ex_partial.push_back(num_block_ht_result_tiles);
    cb_ex_partial.wait_front(num_block_ht_result_tiles);

    // ---------------------------------------------------------------------------
    // Combine Welford local partials with external partials
    // cb_ex_id <-- cb_ex_external_id, cb_ex_partial_id
    // If reduction is single-stage, or this core is a second-stage reader,
    // then cb_ex_id contains mean and 1/sqrt(var + eps) interleaved.
    // Otherwise, cb_ex_id contains mean and var interleaved.
    // ---------------------------------------------------------------------------
    reconfig_data_format_srca(cb_ex_partial_id);
    if constexpr (is_allgather_worker) {
        cb_ex.reserve_back(2 * num_tiles_per_allgather_worker);
        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            norm::kernel_util::compute::combine_welford_partials(
                cb_ex_external,
                cb_ex,
                num_blocks_combine,
                [is_second_stage_reader, num_blocks_first_stage, own_row, boundary_width_index, block_w, last_block_w](
                    uint32_t b) {
                    return get_next_set_size(
                        b,
                        is_second_stage_reader,
                        num_blocks_first_stage,
                        own_row,
                        boundary_width_index,
                        block_w,
                        last_block_w);
                },
                norm::kernel_util::compute::RSqrtPolicy{!(use_two_stage_reduce && !is_second_stage_reader), eps});

            // Just needed to stay in sync with the readers
            if (use_two_stage_reduce && !is_second_stage_reader) {
                // Number of second-stage tiles = 2 * (num_blocks_second_stage - 1)
                // The -1 is the account for the row-column overlap core
                // between first stage (row) and second stage (column) (if row major).
                // The factor of 2 is because each block has 2 tiles (mean, var).
                constexpr uint32_t num_second_stage_tiles = 2 * (num_blocks_second_stage - 1);
                cb_ex_external.wait_front(num_second_stage_tiles);
                cb_ex_external.pop_front(num_second_stage_tiles);
            }
        }
        cb_ex.push_back(2 * num_tiles_per_allgather_worker);
        cb_ex.wait_front(2 * num_tiles_per_allgather_worker);
    }

    // ---------------------------------------------------------------------------
    // Receive the global reduce result and transpose back to columns
    // ---------------------------------------------------------------------------
    cb_ex_global.wait_front(num_block_ht_result_tiles);
    cb_transpose.reserve_back(num_block_ht_result_tiles);
    transpose_init(cb_ex_global_id);
    uint32_t processed_tiles = 0;
    while (processed_tiles < num_block_ht_result_tiles) {
        uint32_t tiles_to_load = std::min(num_block_ht_result_tiles - processed_tiles, num_dest_regs);
        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_to_load; i++) {
            transpose_tile(cb_ex_global_id, processed_tiles + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < tiles_to_load; i++) {
            pack_tile(i, cb_transpose_id);
        }
        tile_regs_release();
        processed_tiles += tiles_to_load;
    }
    cb_transpose.push_back(num_block_ht_result_tiles);
    cb_ex_global.pop_front(num_block_ht_result_tiles);

    cb_transpose.wait_front(num_block_ht_result_tiles);

    // ---------------------------------------------------------------------------
    // Compute x - E[x]
    // ---------------------------------------------------------------------------
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_in_id, cb_transpose_id);
    }
    index_h_offset = 0;
    sub_bcast_cols_init_short(cb_in_id, cb_transpose_id);
    cb_xmm.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_ht; i++) {
        index_subblock_w_offset = 0;
        const auto mean_idx = 2 * i;
        cb_transpose.wait_front(mean_idx + 1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset;
                sub_tiles_bcast_cols(cb_in_id, cb_transpose_id, index, mean_idx, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                pack_tile(sbi, cb_xmm_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_wt;
        }
        cb_in.pop_front(block_wt);
        // Don't pop transpose buffer until after the mul below
    }
    cb_xmm.push_back(num_tiles_per_block);
#ifndef FUSE_PRE_ADD
    reconfig_data_format_srca(cb_in_id, cb_xmm_id);
#endif
    cb_xmm.wait_front(num_tiles_per_block);

    if constexpr (do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(cb_out_id);
    }

    // ---------------------------------------------------------------------------
    // Scale by 1/sqrt(Var[x] + eps)
    // ---------------------------------------------------------------------------
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm_id, cb_transpose_id);
    }
    mul_bcast_cols_init_short(cb_xmm_id, cb_transpose_id);
    index_h_offset = 0;
    cb_im.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_ht; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles_bcast_cols(cb_xmm_id, cb_transpose_id, index, /*1/sqrt(var+eps) idx*/ 1, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                pack_tile(sbi, cb_im_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_wt;
        }
        index_h_offset += block_wt;
        cb_transpose.pop_front(2);
    }
    cb_im.push_back(num_tiles_per_block);
    cb_xmm.pop_front(num_tiles_per_block);

    // ---------------------------------------------------------------------------
    // Scale by gamma
    // ---------------------------------------------------------------------------
    cb_im.wait_front(num_tiles_per_block);
    if constexpr (do_gamma) {
        reconfig_data_format(cb_im_id, cb_gamma_id);
        if constexpr (do_beta == 0) {
            pack_reconfig_data_format(cb_out_id);
        }
        mul_bcast_rows_init_short(cb_im_id, cb_gamma_id);
        cb_gamma.wait_front(block_wt);
        index_h_offset = 0;
        cb_outgamma.reserve_back(num_tiles_per_block);
        for (uint32_t i = 0; i < block_ht; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_wt; w++) {
                    index = w + index_subblock_w_offset;
                    mul_tiles_bcast_rows(cb_im_id, cb_gamma_id, index + index_h_offset, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                    pack_tile(sbi, cb_outgamma_id);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_wt;
            }
            index_h_offset += block_wt;
        }
        cb_outgamma.push_back(num_tiles_per_block);
        cb_im.pop_front(num_tiles_per_block);
        cb_outgamma.wait_front(num_tiles_per_block);
    }

    // ---------------------------------------------------------------------------
    // Add beta
    // ---------------------------------------------------------------------------
    if constexpr (do_beta) {
        reconfig_data_format(cb_fusion_id, cb_beta_id);
        pack_reconfig_data_format(cb_out_id);
        add_bcast_rows_init_short(cb_fusion_id, cb_beta_id);
        cb_beta.wait_front(block_wt);
        index_h_offset = 0;
        cb_out.reserve_back(num_tiles_per_block);
        for (uint32_t i = 0; i < block_ht; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_wt; w++) {
                    index = w + index_subblock_w_offset;
                    add_tiles_bcast_rows(cb_fusion_id, cb_beta_id, index + index_h_offset, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                    pack_tile(sbi, cb_out_id);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_wt;
            }
            index_h_offset += block_wt;
        }
        cb_out.push_back(num_tiles_per_block);
        cb_fusion.pop_front(num_tiles_per_block);
        cb_out.wait_front(num_tiles_per_block);
    }
}
