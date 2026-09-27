// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/activation_types.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul/detail/sfpu_activation_helpers.hpp"

using compute_kernel_lib::ActivationThread;
using compute_kernel_lib::KernelActivation;

namespace compute_kernel_lib {
namespace matmul_config {

// WaitAndRetainOnLastBlock leaves the last K block for caller reuse.
enum class InputPolicy : uint8_t { WaitAndPopPerKBlock, WaitAndRetainOnLastBlock };

// Controls entry initialization; both modes restore matmul state after partial reloads.
enum class InitMode : uint8_t { Initialize, AssumeInitialized };

// Configure input and accumulation pack formats, or retain caller-provided state.
enum class DataFormatReconfig : uint8_t { InputAndOutput, None };

}  // namespace matmul_config

// Matmul owns activation ordering, including the optional post-matmul bias pass.
// Defaults to the pack thread so SFPU activation can overlap math-thread matmul.
template <
    KernelActivation Act = KernelActivation::NONE,
    uint32_t Param0 = 0,
    uint32_t Param1 = 0,
    uint32_t Param2 = 0,
    bool PackRelu = false,
    ActivationThread Thread = ActivationThread::Pack>
struct MatmulActivation {
    static_assert(!PackRelu || Thread == ActivationThread::Pack, "Packer ReLU must run on the packer thread");
    static constexpr bool pack_relu = PackRelu;
    static constexpr bool enabled = PackRelu || Act != KernelActivation::NONE;

    // Call once after compute startup. Math activation runs before DST commit;
    // packer activation replaces tile_regs_wait after commit.
    FORCE_INLINE static void init();
    FORCE_INLINE static void before_commit(uint32_t num_tiles);
    FORCE_INLINE static void after_commit(uint32_t num_tiles);
};

/**
 * Block dimensions in tiles: M = in0_num_subblocks * out_subblock_h,
 * N = in1_num_subblocks * out_subblock_w, K = num_k_blocks * in0_block_k.
 */
struct MatmulShape {
    uint32_t in0_num_subblocks;  // Output subblock count along M.
    uint32_t in1_num_subblocks;  // Output subblock count along N.
    uint32_t out_subblock_h;     // Output subblock height in tiles.
    uint32_t out_subblock_w;     // Output subblock width in tiles.
    uint32_t in0_block_k;        // Tiles per K block.
    uint32_t num_k_blocks;       // K block count.
    uint32_t batch = 1;          // Independent batch slices.

    // Valid columns in the last in1 subblock; 0 uses out_subblock_w. Packing stays full-width.
    uint32_t last_in1_subblock_w_valid = 0;

    // Producer N stride in tiles; 0 uses in1_num_subblocks * out_subblock_w.
    uint32_t in1_per_core_w = 0;

    // UnpackToDestFp32 reload view; UINT32_MAX uses interm.
    uint32_t partials_reload_cb_id = UINT32_MAX;

    // External output sharing partials storage. With bias and software spills,
    // reserve a full block here before overwriting partials. UINT32_MAX disables
    // this extra guard; the ordinary unbiased output guard is always retained.
    uint32_t partials_alias_output_cb_id = UINT32_MAX;

    static constexpr MatmulShape of(
        uint32_t in0_num_subblocks,
        uint32_t in1_num_subblocks,
        uint32_t out_subblock_h,
        uint32_t out_subblock_w,
        uint32_t in0_block_k,
        uint32_t num_k_blocks,
        uint32_t batch = 1,
        uint32_t in1_per_core_w = 0) {
        return {
            in0_num_subblocks,
            in1_num_subblocks,
            out_subblock_h,
            out_subblock_w,
            in0_block_k,
            num_k_blocks,
            batch,
            /*last_in1_subblock_w_valid=*/0,
            in1_per_core_w};
    }
};

// Compile-time counterpart of MatmulShape.
template <
    uint32_t In0NumSubblocks,
    uint32_t In1NumSubblocks,
    uint32_t OutSubblockH,
    uint32_t OutSubblockW,
    uint32_t In0BlockK,
    uint32_t NumKBlocks,
    uint32_t Batch = 1,
    uint32_t LastIn1SubblockWValid = 0,
    uint32_t In1PerCoreW = 0>
struct StaticMatmulShape {
    static constexpr uint32_t in0_num_subblocks = In0NumSubblocks;
    static constexpr uint32_t in1_num_subblocks = In1NumSubblocks;
    static constexpr uint32_t out_subblock_h = OutSubblockH;
    static constexpr uint32_t out_subblock_w = OutSubblockW;
    static constexpr uint32_t in0_block_k = In0BlockK;
    static constexpr uint32_t num_k_blocks = NumKBlocks;
    static constexpr uint32_t batch = Batch;
    static constexpr uint32_t last_in1_subblock_w_valid = LastIn1SubblockWValid;
    static constexpr uint32_t in1_per_core_w = In1PerCoreW;
    uint32_t partials_reload_cb_id = UINT32_MAX;
    uint32_t partials_alias_output_cb_id = UINT32_MAX;
};

struct NoPreKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

// Bias is applied to the completed matmul block before activation. The bias
// buffer is retained by the caller and may be reused for subsequent blocks.
struct MatmulBias {
    uint32_t cb_id = 0;
    uint32_t num_tiles = 0;  // Resident prefix to wait for; includes all tiles addressable by offset.
    uint32_t offset = 0;     // First bias tile for this output block within the resident prefix.
};

enum class MatmulBiasMode : uint8_t {
    RowBroadcast,          // One bias tile per N column, broadcast across tile rows.
    ColumnIndexed,         // One bias tile per N column, added elementwise to each output tile.
    FullBlockElementwise,  // One bias tile per (M, N) output tile.
};

// Describes the operand formats left by matmul. These explicit transitions do
// not consume buffers or run automatically on return/destruction. Keeping them
// at the caller's output boundary avoids restoring formats just before untilize.
template <bool WithBias>
struct MatmulResult {
    uint32_t in0_cb_id;
    uint32_t in1_cb_id;
    uint32_t interm_cb_id;
    uint32_t bias_cb_id;

    // Configure SrcA for untilizing the completed block in interm. The caller
    // still configures the packer for its output CB and initializes untilize.
    FORCE_INLINE void prepare_untilize_input() const;

    // Restore matmul operand formats after bias and/or untilize when another
    // output block follows. This does not initialize the matmul LLK operation.
    FORCE_INLINE void restore_input_formats() const;
};

/**
 * Subblocked matmul with software or packer L1 accumulation.
 * Shape accepts MatmulShape or StaticMatmulShape.
 *
 * Call compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) once at startup.
 * AssumeInitialized also requires matmul_block_init with matching operands and shape.
 * Initialize the activation at startup with Activation::init().
 *
 * Pass CB IDs. in0, in1, and out must be distinct; each K block consumes
 * M x block_k input tiles from in0 and block_k x N from in1, subject to InputPolicy.
 * out and interm may share compatible L1 storage. The same CB ID is allowed only
 * for a kernel-local result with no concurrent consumer. L1 accumulation requires
 * exactly one output block of partials capacity. For one K block, interm may be out.
 * With bias and software spills, set shape.partials_alias_output_cb_id when
 * partials share storage with a concurrently drained output. This may be a
 * different CB from out when the tiled result is subsequently untilized.
 *
 * Output is contiguous within each subblock, in subblock traversal order.
 * Use reblock_and_untilize for untilized output. TransposeIn1 transposes B tiles,
 * not the tile grid, which the caller must arrange.
 *
 * Activation runs on completed sums: ActivationThread::Math runs before DST commit,
 * while ActivationThread::Pack runs on the packer thread. PackRelu uses
 * hardware packer ReLU. WithBias enables bias before activation; pass MatmulBias
 * with a valid CB and tile count. BiasMode selects how bias tiles are indexed.
 * Completed sums are first written to interm, then consumed by the bias pass.
 *
 * PreKBlockFn(block, num_k_blocks, is_last) runs before input waits and must
 * restore matmul state after preprocessing.
 * Bias is currently supported for one batch. FullBlockElementwise uses
 * in1_per_core_w as its N stride and redirects padded columns in the last
 * subblock to the first bias tile; the writer discards those output tiles.
 * The bias buffer is waited on but retained for caller reuse.
 * The returned MatmulResult handles the operand-format transitions into
 * untilize and back to matmul; neither transition is performed automatically.
 *
 * Avoid HiFi4 with BF16 inputs and FP32 DST on Wormhole B0 (issue #38306).
 * SKIP_COMPUTE skips the matmul LLK call but retains synchronization.
 */
template <
    bool TransposeIn1 = false,
    bool PackerL1Acc = false,
    matmul_config::InitMode InitMode = matmul_config::InitMode::Initialize,
    matmul_config::InputPolicy InputPolicy = matmul_config::InputPolicy::WaitAndPopPerKBlock,
    matmul_config::DataFormatReconfig Reconfig = matmul_config::DataFormatReconfig::InputAndOutput,
    typename Activation = MatmulActivation<>,
    bool WithBias = false,
    MatmulBiasMode BiasMode = MatmulBiasMode::RowBroadcast,
    typename PreKBlockFn = NoPreKBlock,
    typename Shape>
ALWI MatmulResult<WithBias> matmul(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t out_cb_id,
    uint32_t interm_cb_id,
    const Shape& shape,
    PreKBlockFn pre_k_block = {},
    MatmulBias bias = {});

}  // namespace compute_kernel_lib

#include "matmul.inl"
