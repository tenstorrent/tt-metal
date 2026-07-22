// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/welford.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/transpose.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/circular_buffer.h"

namespace generic = norm::kernel_util::generic;

template <
    uint32_t cb_in,
    uint32_t cb_inb,
    uint32_t cb_interm_pre_add,
    uint32_t cb_ex,
    uint32_t cb_ex2,
    uint32_t cb_ex_welford,
    uint32_t cb_ex2_welford,
    bool welford_state_fp32_alias,
    uint32_t input_dst,
    uint32_t mean_dst,
    uint32_t var_dst,
    uint32_t Wt,
    uint32_t tile_width,
    uint32_t W,
    uint32_t blk>
void welford_fuse_pre_add(const std::array<uint32_t, W>& reciprocal_lut) {
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_inb_obj(cb_inb);
    CircularBuffer cb_interm_pre_add_obj(cb_interm_pre_add);
    CircularBuffer cb_ex_obj(cb_ex);
    CircularBuffer cb_ex2_obj(cb_ex2);
    // When welford_state_fp32_alias is true these are c_30/c_31; distinct buffer indices
    // sharing cb_ex/cb_ex2's SRAM allocations but configured with UnpackToDestFp32.
    // When false, cb_ex_welford == cb_ex and cb_ex2_welford == cb_ex2.
    CircularBuffer cb_ex_welford_obj(cb_ex_welford);
    CircularBuffer cb_ex2_welford_obj(cb_ex2_welford);

    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    // When last tile is full the value is 0 and is not used because full update is done.
    constexpr uint32_t last_tile_rows = W % tile_width;
    constexpr bool is_last_tile_full = (last_tile_rows == 0);

    uint32_t sample_idx = 0;

    tile_regs_acquire();
    welford_init();
    welford_save_state(mean_dst);
    tile_regs_commit();

    cb_ex_obj.reserve_back(1);
    cb_ex2_obj.reserve_back(1);
    if constexpr (welford_state_fp32_alias) {
        // Must be done in compute: cb_ex / cb_ex2 hold welford state (mean / M2) which are
        // produced by pack_tile below; the reader never writes these CBs. Aliases share SRAM
        // but have independent read/write counters and need to be kept in sync so the next
        // block's wait_front on the aliases (used by copy_tile for fp32 precision) sees the data.
        cb_ex_welford_obj.reserve_back(1);
        cb_ex2_welford_obj.reserve_back(1);
    }
    tile_regs_wait();
    pack_reconfig_data_format(cb_ex);
    pack_tile(mean_dst, cb_ex);
    pack_tile(var_dst, cb_ex2);
    tile_regs_release();
    cb_ex_obj.push_back(1);
    cb_ex2_obj.push_back(1);
    if constexpr (welford_state_fp32_alias) {
        cb_ex_welford_obj.push_back(1);
        cb_ex2_welford_obj.push_back(1);
    }

    for (auto block : generic::blocks(Wt, blk)) {
        // Fused pre-add
        reconfig_data_format(cb_in, cb_inb);
        add_init(cb_in, cb_inb);
        cb_in_obj.wait_front(block.full_block_size());
        cb_inb_obj.wait_front(block.full_block_size());
        tile_regs_acquire();
        for (auto i : block.local()) {
            add_tiles(cb_in, cb_inb, i, i, i);
        }
        tile_regs_commit();
        cb_in_obj.pop_front(block.full_block_size());
        cb_inb_obj.pop_front(block.full_block_size());

        // Pack to intermediate CB (needed
        // to workaround transpose_dest bug)
        pack_reconfig_data_format(cb_interm_pre_add);
        cb_interm_pre_add_obj.reserve_back(block.full_block_size());
        tile_regs_wait();
        for (auto i : block.local()) {
            pack_tile(i, cb_interm_pre_add);
        }
        tile_regs_release();
        cb_interm_pre_add_obj.push_back(block.full_block_size());

        // Now run Welfords in these blk number of tiles
        cb_interm_pre_add_obj.wait_front(block.full_block_size());
        cb_ex_obj.wait_front(1);
        cb_ex2_obj.wait_front(1);
        if constexpr (welford_state_fp32_alias) {
            cb_ex_welford_obj.wait_front(1);
            cb_ex2_welford_obj.wait_front(1);
        }
        tile_regs_acquire();
        // Reload running mean/M2 from the aliases. With welford_state_fp32_alias active
        // these are c_30/c_31 in UnpackToDestFp32 mode so copy_tile takes the Dst path that
        // preserves the full FP32 precision. Otherwise, cb_ex_welford == cb_ex.
        reconfig_data_format_srca(cb_in, cb_ex_welford);
        copy_tile_init(cb_ex_welford);
        copy_tile(cb_ex_welford, 0, mean_dst);
        reconfig_data_format_srca(cb_ex_welford, cb_ex2_welford);
        copy_tile_to_dst_init_short_with_dt(cb_ex_welford, cb_ex2_welford);
        copy_tile(cb_ex2_welford, 0, var_dst);
        welford_restore_state(mean_dst);

        reconfig_data_format_srca(cb_ex2_welford, cb_interm_pre_add);
        transpose_init(cb_interm_pre_add);
        for (auto i : block.local()) {
            // Welford's needs transposed input tile
            transpose_tile(cb_interm_pre_add, i, input_dst);

            // Welford over this tile: include only valid elements, never padding.
            if constexpr (is_last_tile_full) {
                // All tiles can go through the faster call which does 32 rows
                welford_update<W>(input_dst, sample_idx, reciprocal_lut);
            } else {
                // Last tile in width has padding; process only first last_tile_rows rows.
                if ((block.start() + i) == (Wt - 1)) {
                    welford_update_rows<W>(input_dst, sample_idx, 0, last_tile_rows, reciprocal_lut);
                } else {
                    welford_update<W>(input_dst, sample_idx, reciprocal_lut);
                }
            }
            sample_idx += tile_width;
        }
        welford_save_state(mean_dst);
        tile_regs_commit();
        cb_interm_pre_add_obj.pop_front(block.full_block_size());
        cb_ex_obj.pop_front(1);
        cb_ex2_obj.pop_front(1);
        if constexpr (welford_state_fp32_alias) {
            cb_ex_welford_obj.pop_front(1);
            cb_ex2_welford_obj.pop_front(1);
        }

        cb_ex_obj.reserve_back(1);
        cb_ex2_obj.reserve_back(1);
        if constexpr (welford_state_fp32_alias) {
            // This alias update must be in the compute kernel.
            // pack_tile below is the producer of cb_ex / cb_ex2.
            cb_ex_welford_obj.reserve_back(1);
            cb_ex2_welford_obj.reserve_back(1);
        }
        tile_regs_wait();
        pack_reconfig_data_format(cb_interm_pre_add, cb_ex);
        pack_tile(mean_dst, cb_ex);
        pack_tile(var_dst, cb_ex2);
        tile_regs_release();
        cb_ex_obj.push_back(1);
        cb_ex2_obj.push_back(1);
        if constexpr (welford_state_fp32_alias) {
            cb_ex_welford_obj.push_back(1);
            cb_ex2_welford_obj.push_back(1);
        }
    }

    reconfig_data_format_srca(cb_interm_pre_add, cb_ex_welford);

    cb_ex_obj.wait_front(1);
    cb_ex2_obj.wait_front(1);
    if constexpr (welford_state_fp32_alias) {
        cb_ex_welford_obj.wait_front(1);
        cb_ex2_welford_obj.wait_front(1);
    }
    tile_regs_acquire();
    // Final reload before welford_finalize_to_row: same fp32-via-Dst rationale as the
    // per-block reload above.
    copy_tile_init(cb_ex_welford);
    copy_tile(cb_ex_welford, 0, mean_dst);
    copy_tile_to_dst_init_short_with_dt(cb_ex_welford, cb_ex2_welford);
    copy_tile(cb_ex2_welford, 0, var_dst);
    welford_restore_state(mean_dst);
    // Store the mean and variance to the destination registers
    welford_finalize_to_row<W>(mean_dst, W - 1, reciprocal_lut);
    tile_regs_commit();
    cb_ex_obj.pop_front(1);
    cb_ex2_obj.pop_front(1);
    if constexpr (welford_state_fp32_alias) {
        cb_ex_welford_obj.pop_front(1);
        cb_ex2_welford_obj.pop_front(1);
    }
}

/* @brief: Welford's algorithm for no fused pre-add
 * @param: cb_in: input CB
 * @param: input_dst: input tile for Welford's algorithm
 * @param: mean_dst: mean tile for Welford's algorithm
 * @param: Wt: width of the input in tiles
 * @param: tile_width: width of each tile
 * @param: W: width of the input
 * @param: p_reciprocals: pointer to the reciprocal LUT
 */
template <
    uint32_t cb_in,
    uint32_t cb_x_welford,
    bool welford_fp32_alias,
    uint32_t cb_ex,
    uint32_t input_dst,
    uint32_t mean_dst,
    uint32_t Wt,
    uint32_t tile_width,
    uint32_t W,
    uint32_t blk>
void welford_no_fuse_pre_add(const std::array<uint32_t, W>& reciprocal_lut) {
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_x_welford_obj(cb_x_welford);

    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    // When last tile is full the value is 0 and is not used because full update is done.
    constexpr uint32_t last_tile_rows = W % tile_width;
    constexpr bool is_last_tile_full = (last_tile_rows == 0);

    uint32_t sample_idx = 0;
    reconfig_data_format_srca(cb_x_welford);
    // Reconfigure the transpose op for the welford intake CB. When the alias is active,
    // cb_x_welford has UnpackToDestFp32 mode so transpose_tile preserves fp32 precision.
    transpose_init(cb_x_welford);
    tile_regs_acquire();
    welford_init();

    // Process all but the last tile
    for (uint32_t wt = 0; wt < (Wt - 1); ++wt) {
        if constexpr (welford_fp32_alias) {
            cb_x_welford_obj.wait_front(1);
            // SFPU replay slots [0, 32) currently hold the welford recurrence (welford uses the
            // full 32-slot math-thread replay buffer; the recovery block below re-records all
            // of it after each transpose). transpose_init re-records slots [16, 32)
            // with the transpose-dest setup so transpose_tile below can replay them.
            transpose_init(cb_x_welford);
        } else {
            cb_in_obj.wait_front(1);
        }
        // Welford's needs transposed input tile
        transpose_tile(cb_x_welford, 0, input_dst);
        if constexpr (welford_fp32_alias) {
            // transpose_tile took the UnpackToDestFp32 path. Its math-side init clobbered
            // the welford recurrence at SFPU replay slots [16, 32).
            // welford_init<WelfordInitMode::PreserveStats>() re-records all 32 slots with the
            // welford recurrence; PreserveStats keeps the running mean / M2 accumulator in
            // LREG4/5. UNPACK A is left in transpose=1;
            // welford_update is pure SFPU and does not consume that state, and the next
            // iteration's transpose_init reprograms it.
            welford_init<WelfordInitMode::PreserveStats>();
        }
        welford_update<W>(input_dst, sample_idx, reciprocal_lut);

        // Pop the input
        if constexpr (welford_fp32_alias) {
            cb_x_welford_obj.pop_front(1);
        }
        cb_in_obj.pop_front(1);
        sample_idx += tile_width;
    }

    // Process the last tile
    // Reader is sending full blocks, so we need to stay in sync.
    // wait/pop the last tile + any remaining in the last block
    const auto num_to_sync = generic::blocks(Wt, blk).back().remainder() + 1;
    if constexpr (welford_fp32_alias) {
        cb_x_welford_obj.wait_front(num_to_sync);
        transpose_init(cb_x_welford);
    } else {
        cb_in_obj.wait_front(num_to_sync);
    }
    transpose_tile(cb_x_welford, 0, input_dst);
    if constexpr (welford_fp32_alias) {
        welford_init<WelfordInitMode::PreserveStats>();
    }

    if constexpr (is_last_tile_full) {
        welford_update<W>(input_dst, sample_idx, reciprocal_lut);
    } else {
        welford_update_rows<W>(input_dst, sample_idx, 0, last_tile_rows, reciprocal_lut);
    }

    // Store the mean and variance to the destination registers
    welford_finalize_to_row<W>(mean_dst, W - 1, reciprocal_lut);

    tile_regs_commit();

    if constexpr (welford_fp32_alias) {
        cb_x_welford_obj.pop_front(num_to_sync);
    }
    cb_in_obj.pop_front(num_to_sync);
}

void kernel_main() {
    namespace kutil = norm::kernel_util;

    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t do_beta = get_compile_time_arg_val(3);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t W = get_compile_time_arg_val(5);
    constexpr uint32_t tile_width = get_compile_time_arg_val(6);
    constexpr bool fuse_pre_add = static_cast<bool>(get_compile_time_arg_val(8));
    // welford_fp32_alias: when true, cb_x_welford is a multi-buffer-index alias of cb_x
    // configured with UnpackToDestFp32 so the welford section reads full fp32 into DEST
    // while the post-welford eltwise still reads cb_x via SrcA (Tf32).
    // When false, cb_x_welford == cb_x.
    constexpr bool welford_fp32_alias = get_named_compile_time_arg_val("welford_fp32_alias") != 0;
    constexpr auto cb_x_welford = get_named_compile_time_arg_val("cb_x_welford");

    // welford_state_fp32_alias: when true, cb_ex_welford/cb_ex2_welford are c_30/c_31
    // multi-buffer-index aliases of cb_ex (c_18) / cb_ex2 (c_19) configured for UnpackToDestFp32.
    // The fused welford path's per-block copy_tile reads of the running mean / M2 use
    // these aliases to take the Dst fp32 path (preserves FP32 precision) instead of the
    // SrcA Tf32 path. When false, cb_ex_welford == cb_ex and cb_ex2_welford == cb_ex2.
    constexpr bool welford_state_fp32_alias = get_named_compile_time_arg_val("welford_state_fp32_alias") != 0;
    constexpr auto cb_ex_welford = get_named_compile_time_arg_val("cb_ex_welford");
    constexpr auto cb_ex2_welford = get_named_compile_time_arg_val("cb_ex2_welford");

    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    // CB indices - configurable via named compile-time args for kernel chaining support
    constexpr auto cb_eps = get_named_compile_time_arg_val("cb_eps");  // single tile generated by the reader
    constexpr auto cb_in = get_named_compile_time_arg_val("cb_in");    // input x or a for fused pre-add (x=a+b)
    constexpr auto cb_inb = get_named_compile_time_arg_val("cb_inb");  // input b for fused pre-add
    constexpr auto cb_out = get_named_compile_time_arg_val("cb_out");  // output
    constexpr auto cb_gamma = get_named_compile_time_arg_val("cb_gamma");
    constexpr auto cb_beta = get_named_compile_time_arg_val("cb_beta");
    uint32_t cb_xmm = get_named_compile_time_arg_val("cb_xmm");                   // x - E[x]
    constexpr auto cb_ex = get_named_compile_time_arg_val("cb_ex");                    // E[x]
    constexpr auto cb_ex2 = get_named_compile_time_arg_val("cb_ex2");                  // Var[x] = E[(x-E[x])^2]
    constexpr auto cb_ex2pe = get_named_compile_time_arg_val("cb_ex2pe");              // Var[x]+ε
    constexpr auto cb_fusion = get_named_compile_time_arg_val("cb_fusion");            // stream gamma/beta
    constexpr auto cb_interm_pre_add = get_named_compile_time_arg_val("cb_x");         // intermediate for fused pre-add
    constexpr auto cb_reciprocals = get_named_compile_time_arg_val("cb_reciprocals");  // Pre-computed reciprocals

    CircularBuffer cb_eps_obj(cb_eps);
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_inb_obj(cb_inb);
    CircularBuffer cb_out_obj(cb_out);
    CircularBuffer cb_gamma_obj(cb_gamma);
    CircularBuffer cb_beta_obj(cb_beta);
    CircularBuffer cb_ex_obj(cb_ex);
    CircularBuffer cb_ex2_obj(cb_ex2);
    CircularBuffer cb_ex2pe_obj(cb_ex2pe);

    constexpr uint32_t onetile = 1;

    // Initialize the hardware based on the first op
    // that will be done
    if constexpr (fuse_pre_add) {
        // Init for x = in + b
        compute_kernel_hw_startup(cb_in, cb_inb, cb_interm_pre_add);
    } else {
        // Init for transpose
        constexpr auto first_out_cb = cb_ex;
        unary_op_init_common(cb_in, first_out_cb);
    }

    cb_eps_obj.wait_front(onetile);  // comes from the reader

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;  // Input tile for Welford's algorithm
    constexpr uint32_t mean_dst = 1;   // Mean tile for Welford's
    constexpr uint32_t var_dst = 2;    // Variance tile for Welford's

    // Get pointer to the reciprocal LUT
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Depending on whether we need to fuse pre-add, the approach for welford is different.
        // So we move it to a separate function.
        if constexpr (fuse_pre_add) {
            welford_fuse_pre_add<
                cb_in,
                cb_inb,
                cb_interm_pre_add,
                cb_ex,
                cb_ex2,
                cb_ex_welford,
                cb_ex2_welford,
                welford_state_fp32_alias,
                input_dst,
                mean_dst,
                var_dst,
                Wt,
                tile_width,
                W,
                blk>(*p_reciprocals);
        } else {
            welford_no_fuse_pre_add<
                cb_in,
                cb_x_welford,
                welford_fp32_alias,
                cb_ex,
                input_dst,
                mean_dst,
                Wt,
                tile_width,
                W,
                blk>(*p_reciprocals);
        }
        // We should expect that either of the two would have have populated dst regs with mean and
        // variance in mean_dst and var_dst respectively.

        cb_ex_obj.reserve_back(onetile);
        cb_ex2_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_ex);
        pack_tile(mean_dst, cb_ex);
        pack_tile(var_dst, cb_ex2);
        tile_regs_release();
        cb_ex_obj.push_back(onetile);
        cb_ex2_obj.push_back(onetile);

        // Transpose mean and variance back to
        // columns and pack back to CBs
        reconfig_data_format_srca(cb_ex);
        transpose_init(cb_ex);

        cb_ex_obj.wait_front(onetile);
        cb_ex2_obj.wait_front(onetile);
        tile_regs_acquire();
        transpose_tile(cb_ex, 0, mean_dst);
        transpose_tile(cb_ex2, 0, var_dst);
        tile_regs_commit();
        cb_ex_obj.pop_front(onetile);
        cb_ex2_obj.pop_front(onetile);

        cb_ex_obj.reserve_back(onetile);
        cb_ex2_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_ex);
        pack_tile(mean_dst, cb_ex);
        pack_reconfig_data_format(cb_ex2);
        pack_tile(var_dst, cb_ex2);
        tile_regs_release();
        cb_ex_obj.push_back(onetile);
        cb_ex2_obj.push_back(onetile);

        // =====================================
        // Calculate 1/(√(Var(X) + ε))
        // =====================================
        reconfig_data_format(cb_ex2, cb_eps);
        add_init(cb_ex2, cb_eps);

        cb_ex2_obj.wait_front(onetile);
        tile_regs_acquire();
        add_tiles(cb_ex2, cb_eps, 0, 0, dst0);
        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();
        cb_ex2_obj.pop_front(onetile);

        cb_ex2pe_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst0, cb_ex2pe);
        tile_regs_release();
        cb_ex2pe_obj.push_back(onetile);

        // broadcasts the tile since cb_ex2pe is a column vector that contains the important data
        cb_ex2pe_obj.wait_front(onetile);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_ex2pe);
        unary_bcast_init<BroadcastType::COL>(cb_ex2pe, cb_ex2pe);
        unary_bcast<BroadcastType::COL>(cb_ex2pe, 0, dst0);
        cb_ex2pe_obj.pop_front(onetile);
        tile_regs_commit();

        cb_ex2pe_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst0, cb_ex2pe);
        tile_regs_release();
        cb_ex2pe_obj.push_back(onetile);

        // =====================================
        // Second pass over the input.
        // Computes the final value:
        //    x-E[x]
        //(---------------*𝛄)+ß
        //  √(Var(x)+ε)
        // =====================================
        cb_ex2pe_obj.wait_front(onetile);
        cb_ex_obj.wait_front(onetile);

        // Lockstep the cb_x_welford alias's read/write pointers with cb_in's across the eltwise pass.
        // The reader pushes cb_x_welford in pass 2 to match its pass 1 push (see
        // reader_unary_interleaved_ln_large_tensor_welford.cpp); compute pops it here to match
        // cb_in's pop. Both share SRAM but have independent state; popping cb_x_welford keeps it aligned
        // with cb_in so the next NCHt Welford iteration reads from the correct SRAM offset after CB wrap.
        CircularBuffer cb_x_welford_obj_eltwise(cb_x_welford);

        for (auto block : generic::blocks(Wt, blk)) {
            // Last block may only be partially-filled,
            // and only tiles that have data in them are
            // processed, but need to sync with reader on full blocks
            cb_in_obj.wait_front(block.full_block_size());
            if constexpr (welford_fp32_alias && !fuse_pre_add) {
                // cb_x_welford was pushed by the reader in pass 2; wait for the push and pop in
                // lockstep with cb_in. We do not actually read cb_x_welford in the eltwise pass
                // (FPU consumes cb_in via SrcA); this is purely a FIFO-pointer sync.
                cb_x_welford_obj_eltwise.wait_front(block.full_block_size());
            }
            tile_regs_acquire();
            reconfig_data_format(cb_in, cb_ex);
            sub_bcast_cols_init_short(cb_in, cb_ex);
            // x-E[x]
            for (auto i : block.local()) {
                sub_tiles_bcast_cols(cb_in, cb_ex, i, 0, i);
            }
            cb_in_obj.pop_front(block.full_block_size());
            if constexpr (welford_fp32_alias && !fuse_pre_add) {
                cb_x_welford_obj_eltwise.pop_front(block.full_block_size());
            }

            if constexpr (fuse_pre_add) {
                // Fuse in = in + b
                reconfig_data_format_srca(cb_in, cb_inb);
                add_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_inb, cb_inb);
                cb_inb_obj.wait_front(block.full_block_size());
                for (auto i : block.local()) {
                    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                        cb_inb, i, i);
                }
                cb_inb_obj.pop_front(block.full_block_size());
                reconfig_data_format_srca(cb_inb, cb_ex2pe);
            }

            // Multiply by 1/(√(Var(X) + ε)).
            //
            // On Wormhole, binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCB> on c_0 (cb_in)
            // silently corrupts when an earlier unpack op in this kernel routed through an
            // UnpackToDestFp32 CB. The two triggers are different in each path:
            //   * non-fuse welford (welford_fp32_alias): transpose_tile reads cb_x_welford
            //     (c_29, UnpackToDestFp32 alias of cb_x).
            //   * fuse_pre_add welford state (welford_state_fp32_alias): copy_tile reads
            //     cb_ex_welford / cb_ex2_welford (c_30 / c_31, UnpackToDestFp32 aliases of
            //     cb_ex / cb_ex2). cb_interm_pre_add (c_23) is kept in Default mode and the
            //     fuse path's transpose_tile on it does not contribute to the trigger.
            // The leaked unpacker state survives across the welford -> eltwise boundary;
            // even-indexed DEST half blocks accumulate (output = (1+rsqrt)*(x-mean), ~1.286x),
            // odd-indexed blocks produce mostly zeros. The standard reconfig_data_format(...,
            // IGNORE) skip-optimization at the start of the eltwise block does not reset
            // whatever state needs resetting. Blackhole is unaffected.
            //
            // If we're on Wormhole and any UnpackToDestFp32 alias is active in
            // this kernel, stage (x - mean) through cb_xmm and use the mul_tiles_bcast_cols
            // path so the multiply reads through SrcA instead of reusing DEST.
            // In all other cases, use the DEST_TO_SRCB reuse path, to avoid an extra pack/unpack
            // round-trip. Tracked in Issue #45216.
            constexpr bool wh_dest_reuse_workaround_needed =
#if defined(ARCH_WORMHOLE)
                (welford_fp32_alias || welford_state_fp32_alias);
#else
                false;
#endif
            if constexpr (wh_dest_reuse_workaround_needed) {
                tile_regs_commit();

                const uint32_t cb_xmm_intermediate = get_named_compile_time_arg_val("cb_xmm");
                CircularBuffer cb_xmm_intermediate_obj(cb_xmm_intermediate);
                pack_reconfig_data_format(cb_xmm_intermediate);
                cb_xmm_intermediate_obj.reserve_back(block.full_block_size());
                tile_regs_wait();
                for (auto i : block.local()) {
                    pack_tile(i, cb_xmm_intermediate);
                }
                cb_xmm_intermediate_obj.push_back(block.full_block_size());
                tile_regs_release();

                reconfig_data_format(cb_xmm_intermediate, cb_ex2pe);
                mul_bcast_cols_init_short(cb_xmm_intermediate, cb_ex2pe);
                cb_xmm_intermediate_obj.wait_front(block.full_block_size());
                tile_regs_acquire();
                for (auto i : block.local()) {
                    mul_tiles_bcast_cols(cb_xmm_intermediate, cb_ex2pe, i, 0, i);
                }
                cb_xmm_intermediate_obj.pop_front(block.full_block_size());
                tile_regs_commit();
            } else {
                reconfig_data_format_srca(fuse_pre_add ? cb_inb : cb_in, cb_ex2pe);
                mul_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_ex2pe, cb_ex2pe);
                for (auto i : block.local()) {
                    binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                        cb_ex2pe, 0, i);
                }
                tile_regs_commit();
            }

            if constexpr (!(do_gamma == 1 or do_beta == 1)) {
                cb_xmm = cb_out;
            }

            pack_reconfig_data_format(cb_xmm);
            // Sync with writer on full blocks
            CircularBuffer(cb_xmm).reserve_back(block.full_block_size());
            tile_regs_wait();
            for (auto i : block.local()) {
                pack_tile(i, cb_xmm);
            }
            CircularBuffer(cb_xmm).push_back(block.full_block_size());
            tile_regs_release();

            if constexpr (do_gamma == 1) {
                // Multiply by gamma
                reconfig_data_format(cb_xmm, cb_gamma);
                tile_regs_acquire();
                cb_gamma_obj.wait_front(block.full_block_size());
                CircularBuffer(cb_xmm).wait_front(block.full_block_size());
                mul_bcast_rows_init_short(cb_xmm, cb_gamma);
                for (auto i : block.local()) {
                    mul_tiles_bcast_rows(cb_xmm, cb_gamma, i, i, i);
                }
                tile_regs_commit();
                cb_gamma_obj.pop_front(block.full_block_size());
                CircularBuffer(cb_xmm).pop_front(block.full_block_size());

                if constexpr (!do_beta) {
                    pack_reconfig_data_format(cb_out);
                }
                tile_regs_wait();
                if constexpr (!do_beta) {
                    cb_out_obj.reserve_back(block.full_block_size());
                    for (auto i : block.local()) {
                        pack_tile(i, cb_out);
                    }
                    cb_out_obj.push_back(block.full_block_size());
                } else {
                    CircularBuffer(cb_xmm).reserve_back(block.full_block_size());
                    for (auto i : block.local()) {
                        pack_tile(i, cb_xmm);
                    }
                    CircularBuffer(cb_xmm).push_back(block.full_block_size());
                }
                tile_regs_release();
            }

            if constexpr (do_beta == 1) {
                // Add beta
                tile_regs_acquire();
                reconfig_data_format(cb_xmm, cb_beta);
                add_bcast_rows_init_short(cb_xmm, cb_beta);
                CircularBuffer(cb_xmm).wait_front(block.full_block_size());
                cb_beta_obj.wait_front(block.full_block_size());
                for (auto i : block.local()) {
                    add_tiles_bcast_rows(cb_xmm, cb_beta, i, i, i);
                }
                tile_regs_commit();
                cb_beta_obj.pop_front(block.full_block_size());
                CircularBuffer(cb_xmm).pop_front(block.full_block_size());

                pack_reconfig_data_format(cb_out);
                cb_out_obj.reserve_back(block.full_block_size());
                tile_regs_wait();
                for (auto i : block.local()) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                cb_out_obj.push_back(block.full_block_size());
            }
        }

        cb_xmm = get_named_compile_time_arg_val("cb_xmm");  // x minus mean
        cb_ex2pe_obj.pop_front(onetile);
        cb_ex_obj.pop_front(onetile);
    }  // NCHt loop
    // The single eps tile is waited once and reused across all NCHt iterations; pop it at the end
    // so the CB is left balanced.
    cb_eps_obj.pop_front(onetile);
}
