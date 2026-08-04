// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/pack_untilize.h"
#include "api/compute/experimental/fast_untilize.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

// This is the go-to helper for all untilize usage in compute kernels.
// Prefer this over raw pack_untilize_init/pack_untilize_block/pack_untilize_uninit calls.
namespace compute_kernel_lib {

namespace untilize_config {

constexpr uint32_t INVALID_DFB = 0xFFFFFFFFu;

// Register datatype reconfiguration — use when switching data formats between operations.
enum class ReconfigureRegisterDatatypeMode : uint8_t {
    NoReconfigure,            // No reconfiguration
    UnpackReconfigure,        // Reconfigure unpack registers (srcA)
    PackReconfigure,          // Reconfigure pack registers (output)
    UnpackAndPackReconfigure  // Default — reconfigure both unpack and pack
};

// Controls whether pack_untilize_init/pack_untilize_uninit are called.
// When calling untilize() multiple times back-to-back, you can skip redundant
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
    WaitUpfront,  // Wait for all tiles upfront before processing
    NoWait        // Caller manages synchronization externally
};

// Controls whether BH untilize configures DEST remap during init.
// Use AssumeConfigured only when the caller configured remap once before entering a hot loop.
enum class RemapMode : uint8_t {
    Configure,        // Default: init configures remap
    AssumeConfigured  // Caller already configured remap for this kernel
};

}  // namespace untilize_config

// Standalone init/uninit wrappers for manual lifecycle control.
// Prefer using the unified untilize() with InitUninitMode enums instead.
//
// NOTE: When using standalone init/uninit,
// the caller must NOT pass UnpackReconfigure or UnpackAndPackReconfigure to the
// untilize() reconfig_mode — use NoReconfigure or PackReconfigure only.
// If you need unpacker reconfiguration, call unpacker and packer reconfiguration manually
// before untilize_init(), or use the unified untilize() which handles it automatically.
template <
    uint32_t block_width_tiles,
    uint32_t input_dfb,
    uint32_t output_dfb,
    untilize_config::RemapMode remap_mode = untilize_config::RemapMode::Configure>
ALWI void untilize_init();

template <uint32_t block_width_tiles, uint32_t input_dfb, uint32_t output_dfb>
ALWI void untilize_uninit();

/**
 * Untilize: convert tiled data back to row-major format (reverse of tilize).
 *
 * Reads from input CB (tiled), writes to output CB (row-major).
 * Automatically selects single-pass or block-based pack_untilize at compile time
 * based on block_width_tiles vs DEST capacity.
 *
 * PREREQUISITE: Call compute_kernel_hw_startup(input_cb, output_cb) at the
 * start of your kernel before using this function, unless another init or
 * compute_kernel_hw_startup has already been called. The two-argument overload
 * sets srcA=srcB=input_cb. Use the three-argument form
 * compute_kernel_hw_startup(icb0, icb1, ocb) when srcA and srcB differ.
 *
 * ── Template Parameters (compile-time) ──────────────────────────────────────
 *
 *   block_width_tiles — Number of tiles per row (FIRST template param).
 *   input_dfb         — Input DataflowBuffer index (tiled data).
 *   output_dfb        — Output DataflowBuffer index (row-major output, must differ from input_dfb).
 *   init_uninit_mode  — Init/uninit lifecycle control (default: InitAndUninit).
 *   wait_mode         — How to synchronize on input data (default: WaitBlock).
 *   reconfig_mode      — Register datatype reconfiguration (default: UnpackAndPackReconfigure).
 *   remap_mode        — BH DEST remap setup control (default: Configure).
 *                        Configure: helper configures remap during pack untilize init.
 *                        AssumeConfigured: caller already enabled BH DEST remap and no intervening code changes it.
 *
 * ── Block Geometry ─────────────────────────────────────────────────────────
 *
 *   This helper wraps the pack_untilize LLK. Each of the num_blocks
 *   iterations calls the LLK once on a 1×block_width_tiles tile-row
 *   (1 tile tall, block_width_tiles tiles wide).
 *
 *   Total input: block_width_tiles (W) × num_blocks (H) tiles.
 *
 * ── Runtime Parameters ──────────────────────────────────────────────────────
 *
 *   num_blocks          — Number of tile-rows to process (height in tiles).
 *
 * NOTE: Asymmetric DFB page support (total_input_pages) exists only in the tilize helper.
 * The untilize helper always uses symmetric (tile-sized) entries for both input and output DFBs.
 *
 * ── Examples ────────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
 *
 *   // Hardware init — only if no other init or compute_kernel_hw_startup was called before
 *   compute_kernel_hw_startup(dfb_in, dfb_out);
 *
 *   // 1. Basic untilize (most common)
 *   compute_kernel_lib::untilize<4, dfb_in, dfb_out>(num_blocks);
 *
 *   // 2. Wait-upfront (e.g., GroupNorm pattern)
 *   using namespace compute_kernel_lib::untilize_config;
 *   compute_kernel_lib::untilize<10, dfb_in, dfb_out,
 *            InitUninitMode::InitAndUninit,
 *            WaitMode::WaitUpfront>(num_rows);
 *
 *   // 3. Register reconfiguration (switching data formats mid-kernel)
 *   compute_kernel_lib::untilize<4, dfb_in, dfb_out,
 *            untilize_config::InitUninitMode::InitAndUninit,
 *            untilize_config::WaitMode::WaitBlock,
 *            untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(num_blocks);
 *
 *   // 4. Caller-managed synchronization (data already in DFB)
 *   compute_kernel_lib::untilize<4, dfb_in, dfb_out,
 *            untilize_config::InitUninitMode::InitAndUninit,
 *            untilize_config::WaitMode::NoWait>(num_blocks);
 *
 *   // 5. Multiple back-to-back untilize calls — skip redundant init/uninit between them
 *   compute_kernel_lib::untilize<w, dfb_in, dfb_out, untilize_config::InitUninitMode::InitOnly>(blocks);   // first
 *   compute_kernel_lib::untilize<w, dfb_in, dfb_out, untilize_config::InitUninitMode::Neither>(blocks);    // middle
 *   compute_kernel_lib::untilize<w, dfb_in, dfb_out, untilize_config::InitUninitMode::UninitOnly>(blocks); // last
 *
 */
template <
    uint32_t block_width_tiles,
    uint32_t input_dfb,
    uint32_t output_dfb,
    untilize_config::InitUninitMode init_uninit_mode = untilize_config::InitUninitMode::InitAndUninit,
    untilize_config::WaitMode wait_mode = untilize_config::WaitMode::WaitBlock,
    untilize_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
    untilize_config::RemapMode remap_mode = untilize_config::RemapMode::Configure>
ALWI void untilize(uint32_t num_blocks);

}  // namespace compute_kernel_lib

#include "untilize_helpers.inl"
