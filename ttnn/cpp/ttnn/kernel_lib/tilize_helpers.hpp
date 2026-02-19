// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tt-metalium/circular_buffer_constants.h"
#include "api/compute/tilize.h"
#include "api/compute/cb_api.h"
#include "internal/circular_buffer_interface.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

namespace tilize_config {

constexpr uint32_t INVALID_CB = NUM_CIRCULAR_BUFFERS;

// Register datatype reconfiguration — use when switching data formats between operations.
enum class ReconfigureRegisterDatatypeMode : uint8_t {
    NoReconfigure,            // Default — no reconfiguration
    UnpackReconfigure,        // Reconfigure unpack registers (srcA/srcB)
    PackReconfigure,          // Reconfigure pack registers (output)
    UnpackAndPackReconfigure  // Reconfigure both unpack and pack
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

}  // namespace tilize_config

/**
 * Tilize: convert row-major data to tiled format.
 *
 * Reads from input CB (row-major), writes to output CB (tiled).
 * Automatically selects fast_tilize at compile time when hardware supports it
 * (32x32 tiles, Float32/Float16_b format, half-sync dest mode).
 *
 * PREREQUISITE: Call compute_kernel_hw_startup(input_cb, output_cb) at the
 * start of your kernel before using this function. The two-argument overload
 * sets srcA=srcB=input_cb. Use the three-argument form
 * compute_kernel_hw_startup(icb0, icb1, ocb) when srcA and srcB differ.
 *
 * ── Template Parameters (compile-time) ──────────────────────────────────────
 *
 *   input_cb         — Input circular buffer index (0–31, row-major data).
 *   output_cb        — Output circular buffer index (0–31, tiled output, must differ from input_cb).
 *   init_uninit_mode — Init/uninit lifecycle control (default: InitAndUninit).
 *   wait_mode        — How to synchronize on input data (default: WaitBlock).
 *   reconfig_mode    — Register datatype reconfiguration (default: NoReconfigure).
 *
 * ── Runtime Parameters ──────────────────────────────────────────────────────
 *
 *   block_width_tiles  — Number of output tiles per block.
 *   num_blocks         — Number of blocks to process.
 *   total_input_pages  — Total input CB pages across all blocks (default: std::nullopt).
 *       omitted (symmetric): Input and output CBs both have tile-sized pages.
 *                             Each block waits for and pops block_width_tiles pages.
 *       provided (asymmetric): Input CB has non-tile pages (e.g., one page per row).
 *                               Must be > 0. Each block waits for min(32, remaining_pages)
 *                               input pages. Use when the reader produces row-sized pages.
 *
 * ── Examples ────────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
 *
 *   // Hardware init — must come first, pass the input and output CBs
 *   compute_kernel_hw_startup(cb_in, cb_out);
 *
 *   // 1. Basic tilize (most common — symmetric, both CBs have tile-sized pages)
 *   compute_kernel_lib::tilize<cb_in, cb_out>(block_width_tiles, num_blocks);
 *
 *   // 2. Asymmetric — input CB has row-sized pages, output CB has tile-sized pages
 *   compute_kernel_lib::tilize<cb_in, cb_out>(out_tiles_per_block, num_blocks, total_rows);
 *
 *   // 3. Asymmetric, single block — all rows tilized at once
 *   compute_kernel_lib::tilize<cb_in, cb_out>(total_out_tiles, 1, total_rows);
 *
 *   // 4. Register reconfiguration (switching data formats mid-kernel)
 *   using namespace compute_kernel_lib::tilize_config;
 *   compute_kernel_lib::tilize<new_cb, cb_out,
 *          InitUninitMode::InitAndUninit,
 *          WaitMode::WaitBlock,
 *          ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(16, 5);
 *
 *   // 5. Caller-managed synchronization (data already in CB)
 *   compute_kernel_lib::tilize<cb_in, cb_out,
 *          tilize_config::InitUninitMode::InitAndUninit,
 *          tilize_config::WaitMode::NoWait>(block_w, num_blocks);
 *
 *   // 6. Multiple back-to-back tilize calls — skip redundant init/uninit between them
 *   compute_kernel_lib::tilize<cb_in, cb_out, tilize_config::InitUninitMode::InitOnly>(w, blocks);   // first
 *   compute_kernel_lib::tilize<cb_in, cb_out, tilize_config::InitUninitMode::Neither>(w, blocks);    // middle
 *   compute_kernel_lib::tilize<cb_in, cb_out, tilize_config::InitUninitMode::UninitOnly>(w, blocks); // last
 */
template <
    uint32_t input_cb,
    uint32_t output_cb,
    tilize_config::InitUninitMode init_uninit_mode = tilize_config::InitUninitMode::InitAndUninit,
    tilize_config::WaitMode wait_mode = tilize_config::WaitMode::WaitBlock,
    tilize_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>
ALWI void tilize(
    uint32_t block_width_tiles, uint32_t num_blocks, std::optional<uint32_t> total_input_pages = std::nullopt);

}  // namespace compute_kernel_lib

#include "tilize_helpers.inl"
