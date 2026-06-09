// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "api/compute/tilize.h"
#include "api/compute/cb_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

// This is the go-to helper for all tilize usage in compute kernels.
// Prefer this over raw tilize_init/tilize_block/tilize_uninit calls.
namespace compute_kernel_lib {

namespace tilize_config {

constexpr uint32_t INVALID_DFB = 0xFFFFFFFFu;

// Register datatype reconfiguration — use when switching data formats between operations.
enum class ReconfigureRegisterDatatypeMode : uint8_t {
    NoReconfigure,            // No reconfiguration
    UnpackReconfigure,        // Reconfigure unpack registers (srcA/srcB)
    PackReconfigure,          // Reconfigure pack registers (output)
    UnpackAndPackReconfigure  // Default — reconfigure both unpack and pack
};

// Controls whether tilize_init/tilize_uninit are called.
// When calling tilize() multiple times back-to-back, you can skip redundant
// init/uninit between calls: use InitOnly on the first call, Neither on
// middle calls, and UninitOnly on the last call.
enum class InitUninitMode : uint8_t {
    InitAndUninit,  // Default — calls both init and uninit (use for single standalone calls)
    InitOnly,       // Calls init only (use as the first of multiple back-to-back calls)
    UninitOnly,     // Calls uninit only (use as the last of multiple back-to-back calls)
    Neither         // Calls neither (use for middle calls between InitOnly and UninitOnly)
};

// Input synchronization strategy.
enum class WaitMode : uint8_t {
    WaitBlock,    // Default — wait for input per block
    WaitUpfront,  // Wait for all input upfront before processing
    NoWait        // Caller manages synchronization externally
};

// Controls whether fast tilize is used for Float32 data.
//
// You almost never want Lossless, even in "max-precision" kernels. Fast tilize
// truncates fp32 → tf32 on the way into Dest, but every FPU consumer downstream
// (matmul, FPU reduce, FPU binary eltwise) re-reads the tiled output through
// SrcA/SrcB, which is tf32-only. So the precision Lossless "preserves" is
// destroyed by the very next FPU op, and the only cost is a slower tilize path.
//
// Lossless is correct ONLY when ALL of these hold:
//   1. The tiled output is consumed exclusively by SFPU ops reading directly
//      from Dest (e.g. SFPU eltwise via copy_tile / Dest-resident SFPU chains).
//      Any FPU op anywhere in the consumer chain — even one matmul or one
//      add_tiles — invalidates the use case.
//   2. fp32_dest_acc_en = true in the ComputeConfig.
//   3. The input CB is configured with UnpackToDestMode::UnpackToDestFp32.
// If any one is missing, prefer Fast.
enum class Fp32Mode : uint8_t {
    Fast,     // Default — uses fast_tilize for fp32 (truncates to tf32 precision).
              // This matches what FPU ops downstream would do anyway, so it is the
              // right default even when the surrounding kernel is "max precision".
    Lossless  // Forces standard tilize path for fp32 data. Rarely useful — see
              // enum comment above. Do not pick this just because the kernel is
              // "max precision"; pick it only when the SFPU-only consumer chain
              // and the two CB/ComputeConfig prerequisites are actually in place.
};

// Controls whether BH fast tilize configures DEST remap during init.
// Use AssumeConfigured only when the caller configured remap once before entering a hot loop.
enum class RemapMode : uint8_t {
    Configure,        // Default: init configures remap when the selected tilize path needs it
    AssumeConfigured  // Caller already configured remap for this kernel
};

}  // namespace tilize_config

/**
 * Tilize: convert row-major data to tiled format.
 *
 * Reads from input CB (row-major), writes to output CB (tiled).
 * Automatically selects fast_tilize at compile time when hardware supports it
 * (32x32 tiles, Float32/Float16_b format, half-sync dest mode).
 *
 * PREREQUISITE: Call compute_kernel_hw_startup(input_cb, output_cb) at the
 * start of your kernel before using this function, unless another init or
 * compute_kernel_hw_startup has already been called. The two-argument overload
 * sets srcA=srcB=input_cb. Use the three-argument form
 * compute_kernel_hw_startup(icb0, icb1, ocb) when srcA and srcB differ.
 *
 * ── Template Parameters (compile-time) ──────────────────────────────────────
 *
 *   block_width_tiles — Number of output tiles per block (width in tiles, FIRST template param).
 *   input_dfb        — Input DataflowBuffer index (row-major data).
 *   output_dfb       — Output DataflowBuffer index (tiled output, must differ from input_dfb).
 *   init_uninit_mode — Init/uninit lifecycle control (default: InitAndUninit).
 *   wait_mode        — How to synchronize on input data (default: WaitBlock).
 *   reconfig_mode     — Register datatype reconfiguration (default: UnpackAndPackReconfigure).
 *   fp32_mode        — Float32 precision control (default: Fast).
 *                       Fast: uses fast_tilize when possible (truncates fp32 → tf32 in Dest).
 *                             This is the right default — every FPU op downstream
 *                             (matmul, FPU reduce, FPU binary eltwise) re-reads through
 *                             SrcA/SrcB and truncates to tf32 anyway. "Max precision"
 *                             kernels still want Fast unless the tiled output flows
 *                             exclusively to SFPU via Dest.
 *                       Lossless: forces standard tilize path. RARELY USEFUL. Only helps
 *                             when ALL of: (a) the tiled output is consumed exclusively
 *                             by SFPU ops reading from Dest (any FPU consumer voids it),
 *                             (b) fp32_dest_acc_en=true, (c) the input CB uses
 *                             UnpackToDestFp32. See the Fp32Mode enum comment.
 *   remap_mode       — BH DEST remap setup control (default: Configure).
 *                       Configure: helper configures remap when the selected tilize path needs it.
 *                       AssumeConfigured: caller already enabled BH DEST remap and no intervening code changes it.
 *
 * ── Block Geometry ─────────────────────────────────────────────────────────
 *
 *   This helper wraps the tilize LLK. Each of the num_blocks iterations
 *   calls the LLK once on a 1×block_width_tiles tile-row (1 tile tall,
 *   block_width_tiles tiles wide).
 *
 *   Total output: block_width_tiles (W) × num_blocks (H) tiles.
 *
 * ── Runtime Parameters ──────────────────────────────────────────────────────
 *
 *   num_blocks         — Number of tile-rows to process (height in tiles).
 *   total_input_pages  — Total input CB pages across all blocks (default: std::nullopt).
 *       omitted (symmetric): Input and output CBs both have tile-sized pages.
 *                             Each block waits for and pops block_width_tiles pages.
 *       provided (asymmetric): Input CB has non-tile pages (e.g., one page per row).
 *                               Must be > 0. Each block waits for min(32, remaining_pages)
 *                               input pages. Use when the reader produces row-sized pages.
 *
 * NOTE: Asymmetric CB page support (total_input_pages) exists only in the tilize helper.
 * The untilize helper always uses symmetric (tile-sized) pages for both input and output CBs.
 *
 * ── Examples ────────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
 *
 *   // Hardware init — must come first, pass the input and output DFBs
 *   compute_kernel_hw_startup(dfb_in, dfb_out);
 *
 *   // 1. Basic tilize (most common — symmetric, both DFBs have tile-sized entries)
 *   compute_kernel_lib::tilize<block_width_tiles, dfb_in, dfb_out>(num_blocks);
 *
 *   // 2. Asymmetric — input DFB has row-sized entries, output DFB has tile-sized entries
 *   compute_kernel_lib::tilize<out_tiles_per_block, dfb_in, dfb_out>(num_blocks, total_rows);
 *
 *   // 3. Asymmetric, single block — all rows tilized at once
 *   compute_kernel_lib::tilize<total_out_tiles, dfb_in, dfb_out>(1, total_rows);
 *
 *   // 4. Register reconfiguration (switching data formats mid-kernel)
 *   using namespace compute_kernel_lib::tilize_config;
 *   compute_kernel_lib::tilize<16, new_dfb, dfb_out,
 *          InitUninitMode::InitAndUninit,
 *          WaitMode::WaitBlock,
 *          ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(5);
 *
 *   // 5. Caller-managed synchronization (data already in DFB)
 *   compute_kernel_lib::tilize<block_w, dfb_in, dfb_out,
 *          tilize_config::InitUninitMode::InitAndUninit,
 *          tilize_config::WaitMode::NoWait>(num_blocks);
 *
 *   // 6. Multiple back-to-back tilize calls — skip redundant init/uninit between them
 *   compute_kernel_lib::tilize<w, dfb_in, dfb_out, tilize_config::InitUninitMode::InitOnly>(blocks);   // first
 *   compute_kernel_lib::tilize<w, dfb_in, dfb_out, tilize_config::InitUninitMode::Neither>(blocks);    // middle
 *   compute_kernel_lib::tilize<w, dfb_in, dfb_out, tilize_config::InitUninitMode::UninitOnly>(blocks); // last
 *
 *   // 7. Lossless fp32 tilize — RARELY USEFUL. Only correct when the tiled
 *   //    output is consumed exclusively by SFPU ops reading from Dest, AND
 *   //    fp32_dest_acc_en=true, AND the input CB uses UnpackToDestFp32. Any
 *   //    FPU consumer (matmul, FPU reduce, FPU binary eltwise) re-truncates
 *   //    fp32 → tf32 on the way back through SrcA/SrcB — so Lossless gains
 *   //    nothing precision-wise and only costs you a slower tilize path.
 *   //    Default (Fast) is the right choice for almost every kernel, including
 *   //    ones prompted to use "max precision" defaults.
 *   compute_kernel_lib::tilize<block_w, dfb_in, dfb_out,
 *          tilize_config::InitUninitMode::InitAndUninit,
 *          tilize_config::WaitMode::WaitBlock,
 *          tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
 *          tilize_config::Fp32Mode::Lossless>(num_blocks);
 */
template <
    uint32_t block_width_tiles,
    uint32_t input_dfb,
    uint32_t output_dfb,
    tilize_config::InitUninitMode init_uninit_mode = tilize_config::InitUninitMode::InitAndUninit,
    tilize_config::WaitMode wait_mode = tilize_config::WaitMode::WaitBlock,
    tilize_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
    tilize_config::Fp32Mode fp32_mode = tilize_config::Fp32Mode::Fast,
    tilize_config::RemapMode remap_mode = tilize_config::RemapMode::Configure>
ALWI void tilize(uint32_t num_blocks, std::optional<uint32_t> total_input_pages = std::nullopt);

}  // namespace compute_kernel_lib

#include "tilize_helpers.inl"
