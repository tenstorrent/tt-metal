// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/matmul/matmul.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul/reblock_untilize_helpers.hpp"

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"  // pack_init: Quasar re-programs the pack BFD on pack-output-operand switch (§7)
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// This is the Metal 2.0 fork of bmm_large_block_zm_fused_bias_activation.cpp, which still sits
// beside it and still serves the matmul factories that have not been ported. Changes to either
// copy should be evaluated for the other until the last legacy consumer migrates and the legacy
// copy is retired.
//
// The binding and argument names below are this fork's interface: every factory that later ports
// onto it inherits them and cannot rename them.
//
// A third copy lives at
// tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation_copy.cpp
// because tests/tt_metal cannot import ttnn. It tracks the legacy copy, not this one: it is driven
// by imperative Metal calls with positional compile-time args, whereas everything here is named.

/**
 * @brief Transposes a block of tiles from one dataflow buffer to another.
 *
 * This function reads a block of tiles from the input dataflow buffer, performs a width-height
 * (WH) transpose on each tile, and writes the transposed tiles to the output dataflow buffer.
 * The operation is performed in blocks of `block_size` tiles for efficiency, with a separate loop
 * at the end to handle any leftover tiles when the total tile count is not divisible by
 * `block_size`. The default block size is 4, since there are guaranteed to be 4 tiles in the dst
 *               regs irrespective of dst sync mode or data format.
 *
 * @tparam in0_block_num_tiles The number of tiles in the block to be transposed.
 * @tparam block_size The number of tiles in each block to be transposed.
 * @param in0_transpose_dfb_id Dataflow buffer to read the original tiles from.
 * @param in0_dfb_id Dataflow buffer to which the transposed tiles are written.
 */
template <uint32_t in0_block_num_tiles, uint32_t block_size = 4>
FORCE_INLINE void transpose_tile_block(uint32_t in0_transpose_dfb_id, uint32_t in0_dfb_id) {
    DataflowBuffer in0_transpose_dfb(in0_transpose_dfb_id);
    DataflowBuffer in0_dfb(in0_dfb_id);
    constexpr uint32_t num_blocks = in0_block_num_tiles / block_size;
    constexpr uint32_t last_block_size = in0_block_num_tiles % block_size;
    // Lets do 2 passes: One loop until last and one last for the left overs
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        in0_transpose_dfb.wait_front(block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            transpose_tile(in0_transpose_dfb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in0_transpose_dfb.pop_front(block_size);

        in0_dfb.reserve_back(block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            pack_tile(tile_idx, in0_dfb_id);
        }
        tile_regs_release();
        in0_dfb.push_back(block_size);
    }

    if constexpr (last_block_size > 0) {
        in0_transpose_dfb.wait_front(last_block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            transpose_tile(in0_transpose_dfb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in0_transpose_dfb.pop_front(last_block_size);

        in0_dfb.reserve_back(last_block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            pack_tile(tile_idx, in0_dfb_id);
        }
        tile_regs_release();
        in0_dfb.push_back(last_block_size);
    }
}

void kernel_main() {
    using namespace compute_kernel_lib;

// RUNTIME ARGS
#ifdef MATMUL_DRAM_SHARDED
    const bool is_worker_core = get_arg(args::is_worker_core) == 1;
    // if not worker core, skip
    if (not is_worker_core) {
        return;
    }
#endif

    constexpr auto in0_block_w = get_arg(args::in0_block_w);              // inner block size in tiles
    constexpr auto in0_num_subblocks = get_arg(args::in0_num_subblocks);  // outer row block size (in inner row blocks)
    constexpr auto in0_block_num_tiles =
        get_arg(args::in0_block_num_tiles);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr auto in1_num_subblocks =
        get_arg(args::in1_num_subblocks);                     // outer column block size (in inner column blocks)
    constexpr auto in1_block_w = get_arg(args::in1_block_w);  // out_subblock_w*in1_num_subblocks
    constexpr auto num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);  // outer inner dim (in inner dim blocks)
    constexpr auto num_blocks_w_dim = get_arg(args::num_blocks_w_dim);          // outer inner dim (in inner dim blocks)
    constexpr auto num_blocks_h_dim = get_arg(args::num_blocks_h_dim);          // outer inner dim (in inner dim blocks)
    constexpr auto out_subblock_h = get_arg(args::out_subblock_h);              // inner row block size in tiles
    constexpr auto out_subblock_w = get_arg(args::out_subblock_w);              // inner column block size in tiles
    constexpr auto batch = get_arg(args::batch);                                // batch dim
    constexpr bool untilize_out = get_arg(args::untilize_out);                  // untilize output
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_arg(args::get_batch_from_reader);
    // Whether in0 arrives needing a tile transpose. This arrives as a define rather than an
    // argument because dfb::in0_transposed is only bound when the transpose is actually wanted:
    // selecting the buffer with a ternary would name-look-up both operands regardless of the
    // condition and fail on the unbound one, so the #ifdef just below drops the token entirely
    // instead. Future factories adopting this fork must keep the #ifdef, not restore a ternary.
#ifdef IN0_TRANSPOSE_TILE
    constexpr bool in0_transpose_tile = true;
#else
    constexpr bool in0_transpose_tile = false;
#endif

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

#ifdef IN0_TRANSPOSE_TILE
    constexpr uint32_t in0_dfb_id = dfb::in0_transposed;
#else
    constexpr uint32_t in0_dfb_id = dfb::in0;
#endif
    constexpr uint32_t in1_dfb_id = dfb::in1;
    constexpr uint32_t out_dfb_id = dfb::out;
    constexpr uint32_t mm_partials_dfb_id = dfb::intermed0;
    // Buffer view the cross-block reload copies through: the UnpackToDestFp32-marked alias of the
    // partials buffer when it is also read as an FPU operand (fused bias), otherwise the partials
    // buffer itself.
#ifdef MM_PARTIALS_RELOAD_ALIAS
    // The partials buffer is also read as an FPU operand (fused bias) and so cannot carry
    // UnpackToDestFp32; the reload instead copies through this alias view of the same SRAM, which
    // does carry the flag.
    constexpr uint32_t mm_partials_reload_dfb_id = dfb::intermed0_reload_alias;
#else
    constexpr uint32_t mm_partials_reload_dfb_id = mm_partials_dfb_id;
#endif
    constexpr uint32_t untilize_mode_out_dfb_id = untilize_out ? mm_partials_dfb_id : out_dfb_id;
    // When in0 needs to be transposed, the original data is read from the in0 buffer
    // (in0_transpose_dfb_id), transposed, and the result is written to the in0_transposed buffer
    // (in0_dfb_id), which is then used as input for the matmul call.
    constexpr uint32_t in0_transpose_dfb_id = dfb::in0;

#ifdef FUSE_BIAS
    constexpr bool with_bias = true;
    constexpr uint32_t bias_dfb_id = dfb::bias;
    constexpr auto bias_ntiles = get_arg(args::bias_ntiles);
    // true: row-0 broadcast ([N] / [...,1,N]); false: elementwise add_tiles (bias has multiple M rows).
    constexpr bool row_broadcast_bias = (bool)get_arg(args::row_broadcast_bias);
    constexpr MatmulBias bias_config{bias_dfb_id, bias_ntiles};
#ifdef BIAS_FULL_BLOCK
    // The bias buffer holds a full [M, N] tile block for matmul_multicore_reuse_optimized;
    // other callers load one bias row and index bias tiles by N only.
    static_assert(!row_broadcast_bias, "BIAS_FULL_BLOCK requires elementwise bias");
    constexpr MatmulBiasMode bias_mode = MatmulBiasMode::FullBlockElementwise;
#else
    constexpr MatmulBiasMode bias_mode =
        row_broadcast_bias ? MatmulBiasMode::RowBroadcast : MatmulBiasMode::ColumnIndexed;
#endif
#else
    constexpr bool with_bias = false;
    constexpr MatmulBias bias_config{};
    constexpr MatmulBiasMode bias_mode = MatmulBiasMode::RowBroadcast;
#endif

    // Number of valid in1 columns in the last in1 subblock. For the DRAM-sharded variant the
    // planner may pad per_core_N_compute beyond per_core_N_in1_sender so that out_subblock_w can be
    // larger; the reader only pushes per_core_N_in1_sender tiles per block into the in1 buffer. To
    // avoid reading those non-existent (padded) in1 tiles, the compute kernel narrows the
    // matmul_block call on the last in1 subblock to last_subblock_w_valid lanes. When no padding
    // occurs this equals out_subblock_w and the original full-width path is preserved.
#ifdef MATMUL_DRAM_SHARDED
    constexpr auto last_subblock_w_valid = get_arg(args::last_subblock_w_valid);
#else
    constexpr uint32_t last_subblock_w_valid = out_subblock_w;
#endif

    // Default activation parameters keep helper instantiations valid when activation is disabled.
#ifdef SFPU_ACTIVATION
    constexpr KernelActivation activation_type = static_cast<KernelActivation>(get_arg(args::activation_type));
    constexpr auto activation_param0 = get_arg(args::activation_param0);
    constexpr auto activation_param1 = get_arg(args::activation_param1);
    constexpr auto activation_param2 = get_arg(args::activation_param2);
#else
    constexpr KernelActivation activation_type = KernelActivation::NONE;
    constexpr uint32_t activation_param0 = 0;
    constexpr uint32_t activation_param1 = 0;
    constexpr uint32_t activation_param2 = 0;
#endif

    // Feature flags
#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool l1_acc =
#ifdef PACKER_L1_ACC
        true;
#else
        false;
#endif

#ifdef PACK_RELU
    constexpr bool matmul_pack_relu = true;
#else
    constexpr bool matmul_pack_relu = false;
#endif

    // Activate the completed tiled result before untilize, which only reblocks and copies tiles.
    using BmmActivation =
        MatmulActivation<activation_type, activation_param0, activation_param1, activation_param2, matmul_pack_relu>;

    constexpr bool multiple_output_blocks = batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1;
    using Shape = StaticMatmulShape<
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_h,
        out_subblock_w,
        in0_block_w,
        num_blocks_inner_dim,
        1,
        last_subblock_w_valid,
        in1_block_w>;
    constexpr Shape shape{
        /*partials_reload_cb_id=*/mm_partials_reload_dfb_id,
        /*partials_alias_output_cb_id=*/out_dfb_id};

#ifdef FUSE_BIAS
    // Construction can synchronize on Quasar; do not construct an unused CB.
    DataflowBuffer bias_dfb(bias_dfb_id);
#endif
    auto prepare_k_block = [&](uint32_t, uint32_t, bool) {
        if constexpr (in0_transpose_tile) {
            reconfig_data_format_srca(in1_dfb_id, in0_transpose_dfb_id);
            transpose_init(in0_transpose_dfb_id);
            PACK((pack_reconfig_data_format(in0_dfb_id)));
#ifdef ARCH_QUASAR
            // Quasar output-buffer switches require updating the pack descriptor.
            pack_init(in0_dfb_id);
#endif
            if constexpr (l1_acc) {
                PACK((llk_pack_reconfig_l1_acc(0)));
            }
            transpose_tile_block<in0_block_num_tiles>(in0_transpose_dfb_id, in0_dfb_id);
            reconfig_data_format_srca(in0_transpose_dfb_id, in1_dfb_id);
            matmul_block_init(in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
            PACK((pack_reconfig_data_format(mm_partials_dfb_id)));
#ifdef ARCH_QUASAR
            pack_init(mm_partials_dfb_id);
#endif
        }
    };

    // Retain matmul state across calls for heterogeneous-tile DRAM-sharded inputs.
    // Initialize SFPU activation once at startup.
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_dfb_id, in1_dfb_id, mm_partials_dfb_id);
    matmul_block_init(in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    BmmActivation::init();

    // Main loop: batch × output blocks
    for (uint32_t b = 0; b < batch; b++) {
        if constexpr (get_batch_from_reader) {
            // Check whether this batch is valid
#ifdef ARCH_QUASAR
            // Quasar has no BRISC->compute mailbox (ckernel::ThreadId::BriscThreadId does not exist), so the
            // reader-driven batch-skip (is_batch_valid) handoff is not wired here. Batch sparsity is therefore
            // unsupported on Quasar until that DM->TRISC handoff exists: reject it at compile time so a Quasar
            // build with get_batch_from_reader=true fails loudly instead of deadlocking compute on input tiles
            // the reader never pushes (its own !ARCH_QUASAR guard skips them) or consuming stale DFB contents.
            // Mirrors the experimental Quasar bmm kernel.
            static_assert(
                !get_batch_from_reader,
                "get_batch_from_reader (batch sparsity) is unsupported on Quasar until the DM->TRISC "
                "is_batch_valid mailbox handoff is implemented; do not enable it on Quasar.");
            const bool is_batch_valid = true;  // unreachable when the static_assert holds (flag is false)
#else
            bool is_batch_valid = false;
            UNPACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            MATH(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            PACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
#endif
            if (!is_batch_valid) {
                continue;
            }
        }

        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                // Reset packer state for this output block
                if constexpr (multiple_output_blocks) {
                    PACK((pack_reconfig_data_format(mm_partials_dfb_id)));
#ifdef ARCH_QUASAR
                    // Restore the partials descriptor after the previous output block.
                    pack_init(mm_partials_dfb_id);
#endif
                }

                // K-loop matmul, bias, and activation share one library call.
                const auto result = compute_kernel_lib::matmul<
                    in1_transpose_tile,
                    l1_acc,
                    matmul_config::InitMode::AssumeInitialized,
                    matmul_config::InputPolicy::WaitAndPopPerKBlock,
                    matmul_config::DataFormatReconfig::None,
                    BmmActivation,
                    with_bias,
                    bias_mode>(
                    in0_dfb_id,
                    in1_dfb_id,
                    untilize_mode_out_dfb_id,
                    mm_partials_dfb_id,
                    shape,
                    prepare_k_block,
                    bias_config);

#ifdef FUSE_BIAS
                // Multiple width blocks stream a fresh bias block for every (b, bh, bw).
                // With one width block, retain bias for reuse across bh/batch iterations.
                if constexpr (num_blocks_w_dim > 1) {
                    bias_dfb.pop_front(bias_ntiles);
                }
#endif

                // Untilize the completed output when requested.
                if constexpr (untilize_out) {
                    result.prepare_untilize_input();
#if defined ARCH_QUASAR && (defined FP32_DEST_ACC_EN || defined PACKER_L1_ACC)
                    // WH/BH pack_untilize_dest_init configures this itself. Quasar
                    // programs the output descriptor but still needs the gasket format.
                    if constexpr (!with_bias) {
                        PACK((pack_reconfig_data_format(out_dfb_id)));
                    }
#endif

                    reblock_and_untilize<
                        out_subblock_w,
                        out_block_w,
                        /*reconfigure=*/false>(in0_num_subblocks, out_subblock_h, mm_partials_dfb_id, out_dfb_id);
                }

                // Reconfigure for next output block
                if constexpr (multiple_output_blocks) {
                    result.restore_input_formats();
                    matmul_block_init(
                        in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        }
    }
#ifdef FUSE_BIAS
    // For num_blocks_w_dim == 1 the reader pushes bias once and the kernel holds it resident,
    // reusing it across all batch/bh/block iterations without popping. Pop it once here, after the
    // last use, so the buffer is balanced. (For num_blocks_w_dim > 1 the per-block pop above already
    // balances each re-pushed bias block.)
    if constexpr (num_blocks_w_dim == 1) {
        bias_dfb.pop_front(bias_ntiles);
    }
#endif
}
