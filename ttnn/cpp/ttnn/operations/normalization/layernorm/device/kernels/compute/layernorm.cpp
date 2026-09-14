// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Standard C++ includes for fixed-width integer types.
#include <cstdint>

// Define the broadcast operation type for the kernel.
// BCAST_LLKOP: Binary operation type (e.g., ELWMUL for element-wise multiplication).
// BCAST_DIM: Broadcast dimension (e.g., COL for column-wise broadcasting).
#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

// Tenstorrent compute kernel API includes.
#include "api/compute/compute_kernel_api.h"  // Core kernel API (e.g., compute_kernel_hw_startup).
#include "api/compute/bcast.h"                // Broadcast operations (e.g., bcast_cols).
#include "api/compute/eltwise_binary.h"       // Element-wise binary operations (e.g., add_tiles, mul_tiles).
#include "api/compute/layernorm.h"            // LayerNorm-specific operations.
#ifdef TILIZE_IN
#include "api/compute/tilize.h"               // Tilize operations (for row-major to tile conversion).
#endif
#ifdef UNTILIZE_OUT
#include "api/compute/pack_untilize.h"        // Untilize operations (for tile to row-major conversion).
#endif

// Tenstorrent Metalium constants and utilities.
#include <tt-metalium/constants.hpp>          // Hardware constants (e.g., TILE_WIDTH).
#include "experimental/kernel_args.h"        // Kernel argument helpers.

// Tenstorrent Neural Networks (ttnn) includes for normalization utilities.
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"  // Numeric utilities (e.g., row_wise_mean).
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h" // Blocked range utilities.
#include "ttnn/operations/normalization/kernel_util/generic/bit.h"         // Bit manipulation utilities.
#include "ttnn/operations/normalization/layernorm/device/kernels/layernorm_scaler_tiles.h" // Scaler tile utilities.
#include "api/compute/eltwise_unary/sfpu_split_includes.h" // SFPU (Special Function Unit) includes.
#include "api/compute/tile_move_copy.h"      // Tile move/copy operations.
#include "api/compute/eltwise_unary/eltwise_unary.h" // Element-wise unary operations.
#include "api/dataflow/dataflow_buffer.h"    // Dataflow buffer (DFB) API.

// Namespace aliases for brevity.
namespace generic = norm::kernel_util::generic;
namespace kutil = norm::kernel_util;
namespace numeric = kutil::compute::numeric;
namespace policies = kutil::compute::policies;

// ============================================================================
// Kernel Main Function
// ============================================================================
void kernel_main() {
    // --- Load Runtime Arguments ---
    // These arguments are passed from the host and control the kernel's behavior.
    const uint32_t NCHt = get_arg(args::NCHt);  // Number of channels (or rows) to process.
    constexpr auto Wt = get_arg(args::Wt);      // Width (number of columns) in tiles.
    constexpr auto block_size = get_arg(args::block_size); // Block size for tiling.
    constexpr auto do_gamma = get_arg(args::do_gamma);     // Whether to apply gamma (scale).
    constexpr auto do_beta = get_arg(args::do_beta);       // Whether to apply beta (shift).
    constexpr bool FLOAT32_DTYPE = get_arg(args::fp32_dest_acc_en) == 1; // Whether to use FP32 for destination accumulator.
    constexpr bool FLOAT32_REDUCTION = get_arg(args::float32_reduction) == 1; // Whether to use FP32 for reduction.
    constexpr bool LEGACY_RSQRT = get_arg(args::legacy_rsqrt) == 1; // Whether to use legacy reciprocal square root.
    constexpr auto W = get_arg(args::W);        // Total width (columns) of the input tensor.
    constexpr auto tile_width = get_arg(args::tile_width); // Width of a tile (default: 32).

    // --- Dataflow Buffer (DFB) Handles ---
    // These are the buffer identifiers for the kernel's inputs/outputs.
    // The kernel never sees buffer indices; it only interacts with named buffers.
    constexpr auto dfb_scaler_id = dfb::scaler;  // Scaler buffer (single tile from reader).
    constexpr auto dfb_eps_id = dfb::eps;        // Epsilon buffer (single tile from reader).
    constexpr auto dfb_in_id = dfb::in;          // Input buffer (x or a for fused pre-add).
#ifdef FUSE_PRE_ADD
    constexpr auto dfb_inb_id = dfb::inb;        // Input buffer b (for fused pre-add: x = a + b).
#endif
    constexpr auto dfb_out_id = dfb::out;        // Output buffer.
#ifdef FUSE_GAMMA
    constexpr auto dfb_gamma_id = dfb::gamma;    // Gamma buffer (scale weights).
#endif
#ifdef FUSE_BETA
    constexpr auto dfb_beta_id = dfb::beta;      // Beta buffer (shift weights).
#endif
#ifndef RMSNORM
    constexpr auto dfb_ex_id = dfb::ex;          // Expected value buffer (E[x]).
#endif
    constexpr auto dfb_ex2_id = dfb::ex2;        // Expected squared deviation buffer (E[(x-E[x])^2]).
    constexpr auto dfb_xmm2_id = dfb::xmm2;      // Intermediate buffer for (x - E[x])^2.
    constexpr auto dfb_ex2pe_id = dfb::ex2pe;    // Buffer for E[(x-E[x])^2] + epsilon.
#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
    constexpr auto dfb_fusion_id = dfb::fusion;  // Intermediate buffer for gamma/beta fusion.
#endif

    // --- Dataflow Buffer (DFB) Objects ---
    // Instantiate DFB objects for each buffer ID.
    DataflowBuffer dfb_eps(dfb_eps_id);
    DataflowBuffer dfb_in(dfb_in_id);
#ifdef FUSE_PRE_ADD
    DataflowBuffer dfb_inb(dfb_inb_id);
#endif
    DataflowBuffer dfb_out(dfb_out_id);
#ifdef FUSE_GAMMA
    DataflowBuffer dfb_gamma(dfb_gamma_id);
#endif
#ifdef FUSE_BETA
    DataflowBuffer dfb_beta(dfb_beta_id);
#endif
#ifndef RMSNORM
    DataflowBuffer dfb_ex(dfb_ex_id);
#endif
    DataflowBuffer dfb_ex2(dfb_ex2_id);
    DataflowBuffer dfb_xmm2(dfb_xmm2_id);
    DataflowBuffer dfb_ex2pe(dfb_ex2pe_id);
#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
    DataflowBuffer dfb_fusion(dfb_fusion_id);
#endif
    DataflowBuffer dfb_scaler(dfb_scaler_id);

    // --- Tilize Input (if enabled) ---
    // Tilize converts row-major input to tiled format for efficient processing.
#ifdef TILIZE_IN
    constexpr auto dfb_in_rm_id = dfb::in_rm;    // Row-major input buffer.
    DataflowBuffer dfb_in_rm(dfb_in_rm_id);
#endif

    // --- x - Mean Buffer (xmm) ---
    // For RMSNorm without fused pre-add, the deviation (x - E[x]) is written back over the input.
    // Otherwise, a separate buffer is used.
#if defined(RMSNORM) && !defined(FUSE_PRE_ADD)
    // For RMSNorm without fused pre-add, reuse dfb_in for xmm.
    constexpr auto dfb_xmm_id = dfb_in_id;
    DataflowBuffer& dfb_xmm = dfb_in;
#else
    // Otherwise, use a dedicated buffer for xmm.
    constexpr auto dfb_xmm_id = dfb::xmm;
    DataflowBuffer dfb_xmm(dfb_xmm_id);
#endif

    // --- Constants for Tile Operations ---
    constexpr int onetile = 1;    // Single tile.
    constexpr int dst0 = 0;       // Destination register 0.
    constexpr int dst1 = 1;       // Destination register 1.
    constexpr auto scaler0 = 0;   // Scaler index 0.

    // --- Input/Output Buffer Aliasing ---
    // The input buffer (dfb_x) is either:
    // - The input itself (dfb_in).
    // - The post-add result (dfb_x) for fused pre-add.
    // - The deviation (dfb_xmm) for RMSNorm without fused pre-add.
#ifdef FUSE_PRE_ADD
#ifdef RMSNORM
    // For fused pre-add + RMSNorm, dfb_x is the same as dfb_xmm.
    constexpr auto dfb_x_id = dfb_xmm_id;
    DataflowBuffer& dfb_x = dfb_xmm;
#else
    // For fused pre-add + LayerNorm, use a dedicated buffer for dfb_x.
    constexpr auto dfb_x_id = dfb::x;
    DataflowBuffer dfb_x(dfb_x_id);
#endif
#else
    // For non-fused pre-add, dfb_x is the same as dfb_in.
    constexpr auto dfb_x_id = dfb_in_id;
    DataflowBuffer& dfb_x = dfb_in;
#endif

    // --- Hardware Startup ---
    // Initialize the hardware for the kernel based on the operation mode.
    // This configures the dataflow and tile registers for the specific operation.
#ifdef TILIZE_IN
    // For tilize input, initialize with row-major input and tiled output.
    compute_kernel_hw_startup(dfb_in_rm_id, dfb_in_rm_id, dfb_in_id);
#elif defined(FUSE_PRE_ADD)
    // For fused pre-add, initialize with input a, input b, and output x.
    compute_kernel_hw_startup(dfb_in_id, dfb_inb_id, dfb_x_id);
#elif defined(RMSNORM)
    // For RMSNorm, initialize with xmm and xmm2 buffers.
    compute_kernel_hw_startup(dfb_xmm_id, dfb_xmm_id, dfb_xmm2_id);
#else
    // For standard LayerNorm, initialize with x, scaler, and ex buffers.
    compute_kernel_hw_startup(dfb_x_id, dfb_scaler_id, dfb_ex_id);
#endif

    // Wait for the epsilon buffer to be ready (sent by the reader).
    dfb_eps.wait_front(1);

    // --- Output Buffer Aliasing ---
    // The normalized tiles are written to:
    // - A fusion buffer (dfb_fusion) if gamma/beta are fused.
    // - The output buffer (dfb_out) otherwise.
#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
    constexpr auto dfb_im_or_out_id = dfb_fusion_id;
    DataflowBuffer& dfb_im_or_out = dfb_fusion;
#else
    constexpr auto dfb_im_or_out_id = dfb_out_id;
    DataflowBuffer& dfb_im_or_out = dfb_out;
#endif

    // --- Total Buffer Size ---
    // Compute the total buffer size in tiles, including remainder tiles.
    const auto total_buffer_size = generic::blocks(Wt, block_size).total_with_remainder();

    // ============================================================================
    // Main Loop: Process Each Channel (NCHt)
    // ============================================================================
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#ifdef TILIZE_IN
        // Convert row-major input to tiled format.
        tilize_all_blocks_to_dfb<block_size>(dfb_in_rm, dfb_in, Wt);
        // Re-initialize binary ops after tilize hardware reconfiguration.
        // TODO: Replace with targeted DST re-arm (issue #52395).
#ifdef FUSE_PRE_ADD
        compute_kernel_hw_startup(dfb_in_id, dfb_inb_id, dfb_x_id);
#elif defined(RMSNORM)
        compute_kernel_hw_startup(dfb_xmm_id, dfb_xmm_id, dfb_xmm2_id);
#else
        compute_kernel_hw_startup(dfb_x_id, dfb_scaler_id, dfb_ex_id);
#endif
#endif

        // ========================================================================
        // Step 1: Fused Pre-Add (if enabled)
        // Compute x = a + b (element-wise addition of input tensors a and b).
        // ========================================================================
#ifdef FUSE_PRE_ADD
        // Reconfigure data formats for input a and b.
        reconfig_data_format(dfb_in_id, dfb_inb_id);
        pack_reconfig_data_format(dfb_x_id);

        // Initialize addition operation.
        add_init(dfb_in_id, dfb_inb_id);

        // Process each block of tiles.
        for (auto block : generic::blocks(Wt, block_size)) {
            // Wait for input a and b to be ready (full block size).
            dfb_in.wait_front(block.full_block_size());
            dfb_inb.wait_front(block.full_block_size());

            // Acquire tile registers for the block.
            tile_regs_acquire();
            for (auto i : block.local()) {
                // Perform element-wise addition: a + b.
                add_tiles(dfb_in_id, dfb_inb_id, i, i, i);
            }
            tile_regs_commit();

            // Pop the processed tiles from input a and b.
            dfb_in.pop_front(block.full_block_size());
            dfb_inb.pop_front(block.full_block_size());

            // Reserve space in the output buffer (dfb_x).
            dfb_x.reserve_back(block.full_block_size());

            // Wait for tile registers to be ready.
            tile_regs_wait();
            for (auto i : block.local()) {
                // Pack the result tile into dfb_x.
                pack_tile(i, dfb_x_id);
            }
            tile_regs_release();

            // Push the result tiles into dfb_x.
            dfb_x.push_back(block.full_block_size());
        }

        // Reconfigure data formats for the next steps.
#ifndef RMSNORM
        reconfig_data_format(dfb_in_id, dfb_x_id, dfb_inb_id, dfb_scaler_id);
#else
        reconfig_data_format(dfb_in_id, dfb_x_id, dfb_inb_id, dfb_x_id);
#endif
#endif

        // ========================================================================
        // Step 2: Mean Reduction (for LayerNorm)
        // Compute E[x] (expected value) via row-wise reduction.
        // ========================================================================
#ifndef RMSNORM
        // Perform row-wise mean reduction:
        // - PoolType::SUM: Sum reduction.
        // - ReduceDim::REDUCE_ROW: Reduce along rows.
        // - FLOAT32_REDUCTION: Use FP32 for reduction if enabled.
        // - FullBlockWithoutPopPolicy: Do not pop tiles during reduction.
        numeric::row_wise_mean<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            FLOAT32_REDUCTION,
            policies::FullBlockWithoutPopPolicy>(
                dfb_x, dfb_scaler, dfb_ex, W, Wt, block_size, tile_width);

        // ========================================================================
        // Step 3: Mean Subtraction
        // Compute xmm = x - E[x] (centered tensor).
        // ========================================================================
        // Reconfigure data formats for x and ex.
        reconfig_data_format(dfb_x_id, dfb_ex_id);

        // Reserve space for the centered tensor (xmm).
        dfb_xmm.reserve_back(static_cast<uint16_t>(total_buffer_size));

        // Initialize subtraction operation (x - E[x]).
        sub_bcast_cols_init(dfb_x_id, dfb_ex_id);

        // Process each block of tiles.
        for (auto block : generic::blocks(Wt, block_size)) {
            // Acquire tile registers.
            tile_regs_acquire();
            for (auto i : block.local()) {
                // Subtract E[x] from x (broadcast E[x] across columns).
                sub_tiles_bcast_cols(dfb_x_id, dfb_ex_id, i, 0, i);
            }
            tile_regs_commit();

            // Pop the processed tiles from x.
            dfb_x.pop_front(static_cast<uint16_t>(block.full_block_size()));

            // Wait for tile registers to be ready.
            tile_regs_wait();
            for (auto i : block.local()) {
                // Pack the result tile into xmm.
                pack_tile(i, dfb_xmm_id);
            }
            tile_regs_release();

            // Push the result tiles into xmm.
            dfb_xmm.push_back(static_cast<uint16_t>(block.full_block_size()));
        }

        // Pop the E[x] tile (no longer needed).
        dfb_ex.pop_front(1);

#ifndef FUSE_PRE_ADD
        // Reconfigure data format for SrcA (used in later steps).
        reconfig_data_format_srca(dfb_x_id, dfb_xmm_id);
#endif
#endif

        // ========================================================================
        // Step 4: Variance Computation
        // Compute (x - E[x])^2 and its expected value E[(x - E[x])^2].
        // ========================================================================
        // Initialize multiplication operation (xmm * xmm).
        mul_init(dfb_xmm_id, dfb_xmm_id);

        // Process each block of tiles.
        for (auto block : generic::blocks(Wt, block_size)) {
#ifndef RMSNORM
            // For LayerNorm, wait for the full block size in xmm.
            dfb_xmm.wait_front(static_cast<uint16_t>(block.start() + block.size()));
#else
            // For RMSNorm, wait for the full block size in xmm.
            dfb_xmm.wait_front(block.start() + block.full_block_size());
#endif

            // Acquire tile registers.
            tile_regs_acquire();
            for (auto i : block.local()) {
                const auto global_i = block.to_global(i);
                // Compute (x - E[x])^2.
                mul_tiles(dfb_xmm_id, dfb_xmm_id, global_i, global_i, i);
            }
            tile_regs_commit();

            // Reserve space for the squared deviation (xmm2).
            dfb_xmm2.reserve_back(static_cast<uint16_t>(block.full_block_size()));

            // Wait for tile registers to be ready.
            tile_regs_wait();
            for (auto i : block.local()) {
                // Pack the result tile into xmm2.
                pack_tile(i, dfb_xmm2_id);
            }
            tile_regs_release();

            // Push the result tiles into xmm2.
            dfb_xmm2.push_back(static_cast<uint16_t>(block.full_block_size()));
        }

        // Reconfigure data formats for the next steps.
#if defined(RMSNORM) && !defined(FUSE_PRE_ADD)
        reconfig_data_format(dfb_xmm_id, dfb_xmm2_id, dfb_xmm_id, dfb_scaler_id);
#endif

        // Perform row-wise mean reduction for variance:
        // - Compute E[(x - E[x])^2] (variance).
        numeric::row_wise_mean<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            FLOAT32_REDUCTION,
            policies::FullBlockWithPopPolicy>(
                dfb_xmm2, dfb_scaler, dfb_ex2, W, Wt, block_size, tile_width);

        // ========================================================================
        // Step 5: Variance + Epsilon
        // Compute Var[x] + epsilon for numerical stability.
        // ========================================================================
        // Wait for the variance tile to be ready.
        dfb_ex2.wait_front(1);

        // Reconfigure data formats for ex2 and eps.
        reconfig_data_format(dfb_ex2_id, dfb_eps_id);

        // Acquire tile registers.
        tile_regs_acquire();
        // Initialize addition: Var[x] + epsilon.
        add_init(dfb_ex2_id, dfb_eps_id);
        add_tiles(dfb_ex2_id, dfb_eps_id, 0, 0, dst0);

        // Compute reciprocal square root: 1 / sqrt(Var[x] + epsilon).
        rsqrt_tile_init<LEGACY_RSQRT>();
        rsqrt_tile<LEGACY_RSQRT>(dst0);
        tile_regs_commit();

        // Pop the variance tile (no longer needed).
        dfb_ex2.pop_front(1);

        // Reserve space for the result (ex2pe = Var[x] + epsilon).
        dfb_ex2pe.reserve_back(1);
        pack_reconfig_data_format(dfb_ex2pe_id);

        // Wait for tile registers to be ready.
        tile_regs_wait();
        pack_tile(dst0, dfb_ex2pe_id);
        tile_regs_release();

        // Push the result tile into ex2pe.
        dfb_ex2pe.push_back(1);

        // ========================================================================
        // Step 6: Normalization and Affine Scaling
        // Compute (x - E[x]) / sqrt(Var[x] + epsilon) * gamma + beta.
        // ========================================================================
        // Wait for the ex2pe tile (1 / sqrt(Var[x] + epsilon)) to be ready.
        dfb_ex2pe.wait_front(1);

        // Process each block of tiles.
        for (auto block : generic::blocks(Wt, block_size)) {
            // Reconfigure data formats for xmm and ex2pe.
            reconfig_data_format(dfb_xmm_id, dfb_ex2pe_id);

            // Configure the output buffer (either fusion or out).
#if !defined(FUSE_GAMMA) && !defined(FUSE_BETA)
            pack_reconfig_data_format(dfb_out_id);
#else
            pack_reconfig_data_format(dfb_fusion_id);
#endif

            // Reserve space for the normalized tiles.
            dfb_im_or_out.reserve_back(static_cast<uint16_t>(block.full_block_size()));

            // Reconfigure SrcA to the deviation buffer's format.
            // This is needed because the previous gamma/beta step may have changed it.
#if defined(RMSNORM) && !defined(FUSE_PRE_ADD)
#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
            reconfig_data_format_srca(dfb_fusion_id, dfb_xmm_id);
#endif
#endif

            // Acquire tile registers.
            tile_regs_acquire();
            // Initialize multiplication: xmm * (1 / sqrt(Var[x] + epsilon)).
            mul_bcast_cols_init(dfb_xmm_id, dfb_ex2pe_id);
            for (auto i : block.local()) {
                // Multiply xmm by the reciprocal square root (broadcast across columns).
                mul_tiles_bcast_cols(dfb_xmm_id, dfb_ex2pe_id, block.to_global(i), 0, i);

                // Apply activation if enabled (e.g., ReLU, GELU).
                // This is only done if gamma/beta are not fused, or if this is the last step.
#ifdef SFPU_OP_INIT_ACTIVATION
                if constexpr (!(do_gamma == 1 || do_beta == 1)) {
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
                }
#endif
            }
            tile_regs_commit();

            // Wait for tile registers to be ready.
            tile_regs_wait();
            for (auto i : block.local()) {
                // Pack the normalized tile into the intermediate or output buffer.
                pack_tile(i, dfb_im_or_out_id);
            }
            tile_regs_release();

            // Push the normalized tiles into the intermediate or output buffer.
            dfb_im_or_out.push_back(static_cast<uint16_t>(block.full_block_size()));

            // Reconfigure SrcA for the next steps (if gamma/beta are fused).
#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
#if defined(RMSNORM) && !defined(FUSE_PRE_ADD)
            reconfig_data_format_srca(dfb_xmm_id, dfb_fusion_id);
#endif
#endif

            // ========================================================================
            // Step 7: Apply Gamma (Scale)
            // Multiply by gamma (element-wise broadcast multiplication).
            // ========================================================================
#ifdef FUSE_GAMMA
            {
                // Configure the output buffer for gamma multiplication.
#ifndef FUSE_BETA
                pack_reconfig_data_format(dfb_out_id);
#endif
                // Reconfigure SrcB to gamma.
                reconfig_data_format_srcb(dfb_ex2pe_id, dfb_gamma_id);

                // Determine the output buffer for gamma multiplication:
                // - If beta is also fused, use the fusion buffer.
                // - Otherwise, use the output buffer.
#ifdef FUSE_BETA
                constexpr auto dfb_outg_id = dfb_fusion_id;
                DataflowBuffer& dfb_outg = dfb_fusion;
#else
                constexpr auto dfb_outg_id = dfb_out_id;
                DataflowBuffer& dfb_outg = dfb_out;
#endif

                // Wait for gamma to be ready (full block size).
                dfb_gamma.wait_front(block.start() + block.full_block_size());
                dfb_fusion.wait_front(block.full_block_size());

                // Acquire tile registers.
                tile_regs_acquire();
                // Initialize multiplication: fusion * gamma (broadcast rows).
                mul_bcast_rows_init(dfb_fusion_id, dfb_gamma_id);
                for (auto i : block.local()) {
                    // Multiply fusion buffer by gamma (broadcast across rows).
                    mul_tiles_bcast_rows(
                        dfb_fusion_id, dfb_gamma_id, i, block.to_global(i), i);

                    // Apply activation if enabled and beta is not fused.
#ifdef SFPU_OP_INIT_ACTIVATION
                    if constexpr (!(do_beta == 1)) {
                        SFPU_OP_INIT_ACTIVATION
                        SFPU_OP_FUNC_ACTIVATION
                    }
#endif
                }
                tile_regs_commit();

                // Pop the processed tiles from the fusion buffer.
                dfb_fusion.pop_front(block.full_block_size());
                // Note: Gamma is not popped because it is reused for all NCHt iterations.

                // Reserve space for the gamma-scaled tiles.
                dfb_outg.reserve_back(block.full_block_size());

                // Wait for tile registers to be ready.
                tile_regs_wait();
                for (auto i : block.local()) {
                    // Pack the gamma-scaled tile into the output buffer.
                    pack_tile(i, dfb_outg_id);
                }
                tile_regs_release();

                // Push the gamma-scaled tiles into the output buffer.
                dfb_outg.push_back(block.full_block_size());
            }
#endif

            // ========================================================================
            // Step 8: Apply Beta (Shift)
            // Add beta (element-wise broadcast addition).
            // ========================================================================
#ifdef FUSE_BETA
            {
                // Configure the output buffer for beta addition.
                pack_reconfig_data_format(dfb_out_id);

                // Reconfigure SrcB to beta.
#ifdef FUSE_GAMMA
                reconfig_data_format_srcb(dfb_gamma_id, dfb_beta_id);
#else
                reconfig_data_format_srcb(dfb_ex2pe_id, dfb_beta_id);
#endif

                // Wait for beta to be ready (full block size).
                dfb_beta.wait_front(block.start() + block.full_block_size());
                dfb_fusion.wait_front(block.full_block_size());

                // Acquire tile registers.
                tile_regs_acquire();
                // Initialize addition: fusion + beta (broadcast rows).
                add_bcast_rows_init(dfb_fusion_id, dfb_beta_id);
                for (auto i : block.local()) {
                    // Add beta to the fusion buffer (broadcast across rows).
                    add_tiles_bcast_rows(
                        dfb_fusion_id, dfb_beta_id, i, block.to_global(i), i);

                    // Apply activation if enabled.
#ifdef SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
#endif
                }
                tile_regs_commit();

                // Pop the processed tiles from the fusion buffer.
                dfb_fusion.pop_front(block.full_block_size());
                // Note: Beta is not popped because it is reused for all NCHt iterations.

                // Reserve space for the final output tiles.
                dfb_out.reserve_back(block.full_block_size());

                // Wait for tile registers to be ready.
                tile_regs_wait();
                for (auto i : block.local()) {
                    // Pack the final tile into the output buffer.
                    pack_tile(i, dfb_out_id);
                }
                tile_regs_release();

                // Push the final tiles into the output buffer.
                dfb_out.push_back(block.full_block_size());
            }
#endif
        }  // End of block loop for normalization.

        // Pop the ex2pe tile (no longer needed).
        dfb_ex2pe.pop_front(1);
        // Pop the xmm tiles (no longer needed).
        dfb_xmm.pop_front(static_cast<uint16_t>(total_buffer_size));

        // ========================================================================
        // Step 9: Untilize Output (if enabled)
        // Convert tiled output back to row-major format.
        // ========================================================================
#ifdef UNTILIZE_OUT
        constexpr auto dfb_out_rm_id = dfb::out_rm;  // Row-major output buffer.
        DataflowBuffer dfb_out_rm(dfb_out_rm_id);
        untilize_all_blocks_from_dfb<block_size>(dfb_out, dfb_out_rm, Wt);
#endif
    }  // End of NCHt loop (channels).

    // ============================================================================
    // Cleanup: Pop Scaler Tiles
    // ============================================================================
    // The reduce scaler is generated once by the reader and reused across all NCHt iterations.
    // It is never popped during the loop, so we pop it here to balance the buffer.
    // The reader pushes a second scaler tile only when the last column tile is partial
    // (W not a multiple of tile_width), matching row_wise_mean's wait count.
    //
    // The reader generates the scalers using tt::constants::TILE_WIDTH, so this kernel must
    // use the same width for the counts to match. Otherwise, dfb_scaler push/pop counts diverge.
    static_assert(
        tile_width == tt::constants::TILE_WIDTH,
        "layernorm reader generates reduce scalers using TILE_WIDTH; compute must use the same tile "
        "width or dfb_scaler push/pop counts diverge (issue #48487)");

    // Compute the number of scaler tiles to pop.
    constexpr uint32_t num_scaler_tiles = norm::layernorm::reduce_scaler_tile_count(W, tile_width);
    dfb_scaler.pop_front(num_scaler_tiles);
}