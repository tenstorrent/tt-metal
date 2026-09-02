// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include <cstdint>
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/layernorm.h"
#include "api/compute/transpose.h"
#include "api/compute/welford.h"
#include "api/compute/eltwise_binary.h"
#include "ttnn/operations/normalization/kernel_util/compute/combine_welford.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

/**
 * @brief This kernel computes layernorm for sharded tensors using
 *        stable two-pass mean and variance calculations
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
 *     the interleaved mean and variance results to the partial buffer.
 *     This produces 1 mean tile and 1 variance tile per tile row
 *   - The reader kernels populate the external buffer with the core's
 *     partial result + the other partial results from cores in the
 *     same core row for each row the core is assigned
 *   - Each core combines all partial results for its row(s) in
 *     the external buffer into the combined buffer. The combined buffer contains 1 mean tile
 *     followed by 1 tile of 1/sqrt(var + eps) for each assigned row
 *   - The core row's sender core (the first column of cores) collects
 *     all of these combined tiles and multicasts to all cores in the row
 *     into the global buffer
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
 *     external buffer. The results in the external buffer are combined
 *     in the same way.
 *   - The final combined results are collected by the sender core
 *     and multicasted to all cores into the global buffer
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
    // An idle core sits in a hole of a non-rectangular shard grid. It carries this program's dataflow
    // buffers so the reduction's multicast has somewhere to land, and does no work of its own, so its
    // whole body is compiled out.
#ifndef IDLE_CORE

    // ============================================================================
    // Kernel setup
    // ============================================================================

    // ---------------------------------------------------------------------------
    // Compile-time arguments
    // ---------------------------------------------------------------------------
    constexpr auto num_blocks_first_stage = get_arg(args::num_blocks_first_stage);
    constexpr auto block_wt = get_arg(args::block_w);
    constexpr auto block_ht_const = get_arg(args::block_h);
    volatile uint32_t block_ht_volatile = get_arg(args::block_h);
    constexpr auto subblock_wt_const = get_arg(args::subblock_w);
    volatile uint32_t subblock_wt_volatile = get_arg(args::subblock_w);
    constexpr auto num_subblocks_w = get_arg(args::num_subblocks_w);
    constexpr auto num_tiles_per_block = get_arg(args::num_tiles_per_block);
    constexpr bool FLOAT32_DTYPE = get_arg(args::float32_dtype) == 1;
    constexpr bool LEGACY_RSQRT = get_arg(args::legacy_rsqrt) == 1;
    constexpr auto num_blocks_second_stage = get_arg(args::num_blocks_second_stage);
    constexpr auto tile_width = get_arg(args::tile_width);
    constexpr auto last_tile_w = get_arg(args::last_tile_w);
    constexpr auto W = get_arg(args::W);
    constexpr auto eps = get_arg(args::eps);
    constexpr auto per_core_recip_lut_size = get_arg(args::per_core_recip_lut_size);
    // Valid (logical) tile count of the final width block: the number of its tiles that hold any
    // logical data, the last of which may be only partially valid. Fewer than block_wt when the
    // logical width does not fill the width blocks evenly (each block spans a whole number of tiles).
    // A partial boundary tile is counted as a valid tile here; its valid-column count is carried
    // separately in last_tile_w and combined into last_block_w.
    // For example, w=96 gives 3 tiles, which sharded on two cores leaves two real tiles on the first
    // core and one real tile plus one padding tile on the second. For w=80 (also 3 tiles), the second
    // core owns last_block_wt = 1 tile that is itself partial (last_tile_w = 16 valid columns) plus
    // one padding tile.
    constexpr auto last_block_wt = get_arg(args::last_block_wt);
    // gamma and beta each gate a buffer that only exists when their tensor was supplied, so the flag
    // has to reach the preprocessor as well as `if constexpr`.
#ifdef FUSE_GAMMA
    constexpr bool do_gamma = true;
#else
    constexpr bool do_gamma = false;
#endif
#ifdef FUSE_BETA
    constexpr bool do_beta = true;
#else
    constexpr bool do_beta = false;
#endif
    // Only the cores that gather read the cross-core combine's arguments and touch its buffers, so the
    // distinction is a compile-time one: their runtime-argument schemas differ.
#ifdef IS_ALLGATHER_WORKER
    constexpr bool is_allgather_worker = true;
#else
    constexpr bool is_allgather_worker = false;
#endif

    // ---------------------------------------------------------------------------
    // Dataflow buffer definitions
    // ---------------------------------------------------------------------------
    constexpr uint32_t dfb_in0 = dfb::in0;
#ifdef FUSE_PRE_ADD
    constexpr uint32_t dfb_in1 = dfb::in1;
#endif
#ifdef FUSE_GAMMA
    constexpr uint32_t dfb_gamma_id = dfb::gamma;
#endif
#ifdef FUSE_BETA
    constexpr uint32_t dfb_beta_id = dfb::beta;
#endif
    constexpr uint32_t dfb_x = dfb::x;                       // x minus mean
    constexpr uint32_t dfb_xmm_id = dfb::xmm;                // x minus mean
    constexpr uint32_t dfb_ex_partial_id = dfb::ex_partial;  // Interleaved E[x] and Var[x] partial results
    constexpr uint32_t dfb_ex_id = dfb::ex;                  // Interleaved E[x] and Var[x] global reduce
    constexpr uint32_t dfb_ex_external_id = dfb::ex_external;
    constexpr uint32_t dfb_ex_global_id = dfb::ex_global;  // Interleaved E[x] and Var[x] final global mcast result
    constexpr uint32_t dfb_transpose_id = dfb::transpose;  // Transpose interleaved E[x] and Var[x] to columns
                                                           // (workaround for bug in transpose_dest)
    constexpr uint32_t dfb_fusion_id = dfb::xmm;           // stream gamma/beta
    constexpr uint32_t dfb_out_id = dfb::out;
    constexpr uint32_t dfb_reciprocals = dfb::reciprocals;  // LUT of pre-computed reciprocals for Welford's algorithm

#ifdef FUSE_GAMMA
    DataflowBuffer dfb_gamma(dfb_gamma_id);
#endif
#ifdef FUSE_BETA
    DataflowBuffer dfb_beta(dfb_beta_id);
#endif
    DataflowBuffer dfb_xmm(dfb_xmm_id);
    DataflowBuffer dfb_ex_partial(dfb_ex_partial_id);
    DataflowBuffer dfb_ex(dfb_ex_id);
    DataflowBuffer dfb_ex_external(dfb_ex_external_id);
    DataflowBuffer dfb_ex_global(dfb_ex_global_id);
    DataflowBuffer dfb_transpose(dfb_transpose_id);
    DataflowBuffer dfb_fusion(dfb_fusion_id);
    DataflowBuffer dfb_out(dfb_out_id);

    constexpr uint32_t dfb_im_id = (do_gamma | do_beta) ? dfb_x : dfb_out_id;
    DataflowBuffer dfb_im(dfb_im_id);
    constexpr uint32_t dfb_outgamma_id = do_beta ? dfb_fusion_id : dfb_out_id;
    DataflowBuffer dfb_outgamma(dfb_outgamma_id);
#ifdef FUSE_PRE_ADD
    constexpr uint32_t dfb_in_id = dfb_x;
#else
    constexpr uint32_t dfb_in_id = dfb_in0;
#endif
    DataflowBuffer dfb_in(dfb_in_id);

    // Welford-fp32 alias of the intake buffer. When the alias is active it is a separate buffer index
    // sharing the intake's SRAM but configured with UnpackToDest, so Welford's transpose_tile preserves
    // fp32 precision in DEST. The two aliased indices have independent read/write pointers so the fused
    // path pushes both side by side; the non-fused path reads the sharded input without read/write
    // pointer manipulation, and so does the alias. When the alias is inactive the name resolves to the
    // intake buffer itself.
#ifdef WELFORD_FP32_ALIAS
    constexpr bool welford_fp32_alias = true;
    constexpr uint32_t dfb_x_welford_id = dfb::x_welford;
#else
    constexpr bool welford_fp32_alias = false;
    constexpr uint32_t dfb_x_welford_id = dfb_in_id;
#endif
    DataflowBuffer dfb_x_welford(dfb_x_welford_id);

    // ---------------------------------------------------------------------------
    // Derived quantities
    // ---------------------------------------------------------------------------
    // set block_ht to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_ht = (block_wt == 1) ? block_ht_volatile : block_ht_const;
    const uint32_t subblock_wt = (block_wt <= 2) ? subblock_wt_volatile : subblock_wt_const;

    // This core's real (logical) column count. This path has no per-column mask, so each core must reduce
    // over exactly its logical columns: cores before the last own a full block_w, and the last real
    // core owns the remaining logical columns; a whole number of full tiles plus, when the logical
    // width is not tile-aligned, a final partial tile. The reduce must stop there rather than at the
    // physical shard end, which carries padding tiles. This is the only per-core quantity that differs
    // for the partial final shard.
    const uint32_t welford_reduce_w = get_arg(args::welford_reduce_w);
    const uint32_t partial_reduce_W = welford_reduce_w;

    // Split the local reduction into full tiles and a final partial tile (present only when the
    // logical width is not a multiple of the tile width).
    const uint32_t num_full_welford_tiles = welford_reduce_w / tile_width;
    const uint32_t partial_welford_tile_w = welford_reduce_w % tile_width;

#ifdef IS_ALLGATHER_WORKER
    // This is the number of tile rows to process
    const uint32_t num_tiles_per_allgather_worker = get_arg(args::num_rows_per_all_to_all_worker);

    // These are for two-stage reductions
    const bool use_two_stage_reduce = get_arg(args::use_two_stage_reduce) == 1;
    const bool is_second_stage_reader = get_arg(args::is_second_stage_reader) == 1;

    // Global width-block index of the partial boundary block and this core's own width-block index,
    // read only on all-to-all workers (the cores that run the cross-core combine). own_row is this
    // core's first-stage row; a width shard's global index is own_row * num_blocks_first_stage + its
    // position within the row. These let the combine weight each block/row by its true logical width.
    const uint32_t boundary_width_index = get_arg(args::boundary_width_index);
    const uint32_t my_width_index = get_arg(args::my_width_index);
#else
    const uint32_t num_tiles_per_allgather_worker = 0;
    const bool use_two_stage_reduce = false;
    const bool is_second_stage_reader = false;
    const uint32_t boundary_width_index = 0;
    const uint32_t my_width_index = 0;
#endif
    const uint32_t own_row = my_width_index / num_blocks_first_stage;

    constexpr uint32_t block_w = block_wt * tile_width;
    // Width (valid columns) of the final width block, weighting it in the cross-core combine. The
    // final block owns last_block_wt tiles (<= block_wt), the last of which has last_tile_w valid
    // columns; the other blocks each own a full block_w.
    constexpr uint32_t last_block_w = (last_block_wt - 1) * tile_width + last_tile_w;

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

    // Statistics destination registers
    constexpr uint32_t welford_input_dst = 0;
    constexpr uint32_t welford_mean_dst = 1;
    constexpr uint32_t welford_var_dst = 2;
    constexpr uint32_t retained_welford_input_dst = 3;

    // Pointer to the reciprocal LUT

    using recip_lut_t = std::array<uint32_t, per_core_recip_lut_size>;
    auto p_reciprocals = norm::kernel_util::compute::memory::get_pointer_to_cb_data<recip_lut_t>(dfb_reciprocals, 0);

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
    compute_kernel_hw_startup(dfb_in0, dfb_in1, dfb_in_id);
#else
    compute_kernel_hw_startup(dfb_in_id, dfb_ex_partial_id);
    copy_init(dfb_in_id);
#endif

    // ---------------------------------------------------------------------------
    // Pre-add x + y
    // ---------------------------------------------------------------------------
#ifdef FUSE_PRE_ADD
    reconfig_data_format_srcb(dfb_in0, dfb_in1);
    add_init(dfb_in0, dfb_in1);
    dfb_in.reserve_back(num_tiles_per_block);
    if constexpr (welford_fp32_alias) {
        // Must be done in the compute kernel: on the fused path compute is the producer of the intake
        // buffer via the add_tiles -> pack_tile sequence below; the reader never writes it.
        // The alias shares the intake's SRAM but has its own read/write pointers, so reserve and push
        // both indices side by side. pack_tile writes once via the intake's wr_ptr; the alias lets the
        // welford section wait_front on it independently of the intake.
        dfb_x_welford.reserve_back(num_tiles_per_block);
    }
    for (uint32_t i = 0; i < block_ht; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                add_tiles(dfb_in0, dfb_in1, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                pack_tile(sbi, dfb_in_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_wt;
        }
        index_h_offset += block_wt;
    }
    dfb_in.push_back(num_tiles_per_block);
    dfb_in.wait_front(num_tiles_per_block);
    if constexpr (welford_fp32_alias) {
        dfb_x_welford.push_back(num_tiles_per_block);
        dfb_x_welford.wait_front(num_tiles_per_block);
    }
#endif

    // ---------------------------------------------------------------------------
    // Compute E[x] and Var[x] using stable two-pass statistics
    // ---------------------------------------------------------------------------
    reconfig_data_format_srca(dfb_x_welford_id);
    dfb_ex_partial.reserve_back(num_block_ht_result_tiles);
    // Reconfigure the transpose op for the statistics intake buffer. When the alias is active,
    // it has UnpackToDest mode so transpose_tile preserves fp32 precision.
    transpose_init(dfb_x_welford_id);
    two_pass_stats_init_shifted();
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_ht; i++) {
        tile_regs_acquire();
        // Retain the first three transposed tiles in otherwise idle DEST slots so the
        // centred second pass can reuse them without another unpack.
        if (num_full_welford_tiles > 0) {
            transpose_tile(dfb_x_welford_id, index_h_offset, retained_welford_input_dst);
            two_pass_stats_update_shifted_rows<false /* accumulate_m2 */, true /* initialize_anchor */>(
                retained_welford_input_dst, 0, tile_width);
        }
        for (uint32_t w = 1; w < num_full_welford_tiles; ++w) {
            const uint32_t stats_input_dst = w < 3 ? w : welford_input_dst;
            transpose_tile(dfb_x_welford_id, w + index_h_offset, stats_input_dst);
            two_pass_stats_update_shifted_rows<false /* accumulate_m2 */>(stats_input_dst, 0, tile_width);
        }
        // Do the partial statistics tile, if any. It is the tile immediately after this core's full tiles
        // (index_h_offset + num_full_welford_tiles), i.e. the last real tile of this core's logical
        // columns; not necessarily the last physical tile of the shard (block_wt - 1), which on the
        // final core is a pure-padding tile when the width is split across cores.
        if (partial_welford_tile_w > 0) {
            const uint32_t stats_input_dst =
                num_full_welford_tiles < 3
                    ? (num_full_welford_tiles == 0 ? retained_welford_input_dst : num_full_welford_tiles)
                    : welford_input_dst;
            transpose_tile(dfb_x_welford_id, index_h_offset + num_full_welford_tiles, stats_input_dst);
            if (num_full_welford_tiles == 0) {
                two_pass_stats_update_shifted_rows<false /* accumulate_m2 */, true /* initialize_anchor */>(
                    stats_input_dst, 0, partial_welford_tile_w);
            } else {
                two_pass_stats_update_shifted_rows<false /* accumulate_m2 */>(
                    stats_input_dst, 0, partial_welford_tile_w);
            }
        }
        two_pass_stats_finish_shifted_mean((*p_reciprocals)[partial_reduce_W - 1]);

        for (uint32_t w = 0; w < num_full_welford_tiles; ++w) {
            const uint32_t stats_input_dst = w < 3 ? (w == 0 ? retained_welford_input_dst : w) : welford_input_dst;
            if (w >= 3) {
                transpose_tile(dfb_x_welford_id, w + index_h_offset, welford_input_dst);
            }
            two_pass_stats_update_rows(stats_input_dst, 0, tile_width);
        }
        if (partial_welford_tile_w > 0) {
            const uint32_t stats_input_dst =
                num_full_welford_tiles < 3
                    ? (num_full_welford_tiles == 0 ? retained_welford_input_dst : num_full_welford_tiles)
                    : welford_input_dst;
            if (num_full_welford_tiles >= 3) {
                transpose_tile(dfb_x_welford_id, index_h_offset + num_full_welford_tiles, welford_input_dst);
            }
            two_pass_stats_update_rows(stats_input_dst, 0, partial_welford_tile_w);
        }
        two_pass_stats_finalize_to_row(welford_mean_dst, (*p_reciprocals)[partial_reduce_W - 1]);
        // We should transpose back to columns here
        // However, transpose_dest() is currently buggy.
        // So we transpose to an intermediate buffer downstream
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(welford_mean_dst, dfb_ex_partial_id);
        pack_tile(welford_var_dst, dfb_ex_partial_id);
        tile_regs_release();
        index_h_offset += block_wt;
    }
    dfb_ex_partial.push_back(num_block_ht_result_tiles);
    dfb_ex_partial.wait_front(num_block_ht_result_tiles);

    // ---------------------------------------------------------------------------
    // Combine Welford local partials with external partials
    // If reduction is single-stage, or this core is a second-stage reader,
    // then the combined buffer contains mean and 1/sqrt(var + eps) interleaved.
    // Otherwise, it contains mean and var interleaved.
    // ---------------------------------------------------------------------------
    reconfig_data_format_srca(dfb_ex_partial_id);
    if constexpr (is_allgather_worker) {
        dfb_ex.reserve_back(2 * num_tiles_per_allgather_worker);
        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            norm::kernel_util::compute::combine_welford_partials(
                dfb_ex_external,
                dfb_ex,
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
                dfb_ex_external.wait_front(static_cast<uint16_t>(num_second_stage_tiles));
                dfb_ex_external.pop_front(static_cast<uint16_t>(num_second_stage_tiles));
            }
        }
        dfb_ex.push_back(2 * num_tiles_per_allgather_worker);
        dfb_ex.wait_front(2 * num_tiles_per_allgather_worker);
    }

    // ---------------------------------------------------------------------------
    // Receive the global reduce result and transpose back to columns
    // ---------------------------------------------------------------------------
    dfb_ex_global.wait_front(num_block_ht_result_tiles);
    dfb_transpose.reserve_back(num_block_ht_result_tiles);
    transpose_init(dfb_ex_global_id);
    uint32_t processed_tiles = 0;
    while (processed_tiles < num_block_ht_result_tiles) {
        uint32_t tiles_to_load = std::min(num_block_ht_result_tiles - processed_tiles, num_dest_regs);
        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_to_load; i++) {
            transpose_tile(dfb_ex_global_id, processed_tiles + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < tiles_to_load; i++) {
            pack_tile(i, dfb_transpose_id);
        }
        tile_regs_release();
        processed_tiles += tiles_to_load;
    }
    dfb_transpose.push_back(num_block_ht_result_tiles);
    dfb_ex_global.pop_front(num_block_ht_result_tiles);

    dfb_transpose.wait_front(num_block_ht_result_tiles);

    // ---------------------------------------------------------------------------
    // Compute x - E[x]
    // ---------------------------------------------------------------------------
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(dfb_in_id, dfb_transpose_id);
    }
    index_h_offset = 0;
    sub_bcast_cols_init(dfb_in_id, dfb_transpose_id);
    dfb_xmm.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_ht; i++) {
        index_subblock_w_offset = 0;
        const auto mean_idx = 2 * i;
        dfb_transpose.wait_front(mean_idx + 1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset;
                sub_tiles_bcast_cols(dfb_in_id, dfb_transpose_id, index, mean_idx, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                pack_tile(sbi, dfb_xmm_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_wt;
        }
        dfb_in.pop_front(block_wt);
        // Don't pop transpose buffer until after the mul below
    }
    dfb_xmm.push_back(num_tiles_per_block);
#ifndef FUSE_PRE_ADD
    reconfig_data_format_srca(dfb_in_id, dfb_xmm_id);
#endif
    dfb_xmm.wait_front(num_tiles_per_block);

    if constexpr (do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(dfb_out_id);
    }

    // ---------------------------------------------------------------------------
    // Scale by 1/sqrt(Var[x] + eps)
    // ---------------------------------------------------------------------------
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(dfb_xmm_id, dfb_transpose_id);
    }
    mul_bcast_cols_init(dfb_xmm_id, dfb_transpose_id);
    index_h_offset = 0;
    dfb_im.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_ht; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles_bcast_cols(dfb_xmm_id, dfb_transpose_id, index, /*1/sqrt(var+eps) idx*/ 1, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                pack_tile(sbi, dfb_im_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_wt;
        }
        index_h_offset += block_wt;
        dfb_transpose.pop_front(2);
    }
    dfb_im.push_back(num_tiles_per_block);
    dfb_xmm.pop_front(num_tiles_per_block);

    // ---------------------------------------------------------------------------
    // Scale by gamma
    // ---------------------------------------------------------------------------
    dfb_im.wait_front(num_tiles_per_block);
#ifdef FUSE_GAMMA
    {
        reconfig_data_format(dfb_im_id, dfb_gamma_id);
        if constexpr (do_beta == 0) {
            pack_reconfig_data_format(dfb_out_id);
        }
        mul_bcast_rows_init(dfb_im_id, dfb_gamma_id);
        dfb_gamma.wait_front(block_wt);
        index_h_offset = 0;
        dfb_outgamma.reserve_back(num_tiles_per_block);
        for (uint32_t i = 0; i < block_ht; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_wt; w++) {
                    index = w + index_subblock_w_offset;
                    mul_tiles_bcast_rows(dfb_im_id, dfb_gamma_id, index + index_h_offset, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                    pack_tile(sbi, dfb_outgamma_id);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_wt;
            }
            index_h_offset += block_wt;
        }
        dfb_outgamma.push_back(num_tiles_per_block);
        dfb_im.pop_front(num_tiles_per_block);
        dfb_outgamma.wait_front(num_tiles_per_block);
    }
#endif

    // ---------------------------------------------------------------------------
    // Add beta
    // ---------------------------------------------------------------------------
#ifdef FUSE_BETA
    {
        reconfig_data_format(dfb_fusion_id, dfb_beta_id);
        pack_reconfig_data_format(dfb_out_id);
        add_bcast_rows_init(dfb_fusion_id, dfb_beta_id);
        dfb_beta.wait_front(block_wt);
        index_h_offset = 0;
        dfb_out.reserve_back(num_tiles_per_block);
        for (uint32_t i = 0; i < block_ht; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_wt; w++) {
                    index = w + index_subblock_w_offset;
                    add_tiles_bcast_rows(dfb_fusion_id, dfb_beta_id, index + index_h_offset, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                    pack_tile(sbi, dfb_out_id);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_wt;
            }
            index_h_offset += block_wt;
        }
        dfb_out.push_back(num_tiles_per_block);
        dfb_fusion.pop_front(num_tiles_per_block);
        dfb_out.wait_front(num_tiles_per_block);
    }
#endif

#endif  // IDLE_CORE
}
