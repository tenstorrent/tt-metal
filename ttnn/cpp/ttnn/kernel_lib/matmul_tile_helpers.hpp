// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

namespace matmul_tile_config {

// Register datatype reconfiguration — use when switching data formats between operations.
enum class ReconfigureRegisterDatatypeMode : uint8_t {
    NoReconfigure,            // No reconfiguration
    UnpackReconfigure,        // Reconfigure unpack registers (srcA and srcB)
    PackReconfigure,          // Reconfigure pack registers (output)
    UnpackAndPackReconfigure  // Default — reconfigure both unpack and pack
};

// Controls whether mm_init is called.
// Note: there is no mm_uninit in the LLK API, so UninitOnly and Neither are both no-ops.
// These modes are included for API symmetry with tilize/untilize helpers.
// When calling matmul_tile() multiple times back-to-back, use InitOnly on the first call,
// Neither on middle calls, and UninitOnly (or Neither) on the last call to avoid
// redundant re-initialization overhead.
enum class InitUninitMode : uint8_t {
    InitAndUninit,  // Default — calls mm_init (uninit is a no-op; included for symmetry)
    InitOnly,       // Calls mm_init only; use as the first of multiple back-to-back calls
    UninitOnly,     // No-op — there is no mm_uninit in the LLK API; included for API symmetry
    Neither         // No-op — skips init entirely; use for middle calls in a chain
};

// Input synchronization strategy.
enum class WaitMode : uint8_t {
    WaitPerTile,  // Default — cb_wait_front for 1 in0 tile + 1 in1 tile per Kt iteration;
                  // tiles are popped immediately after matmul_tiles. CB depth of 1 is sufficient.
    WaitUpfront,  // Wait for all Kt in0 tiles and Kt*Nt in1 tiles before the Nt loop,
                  // then pop them all after all Nt output tiles for that Mt row are packed.
                  // Use when the reader pre-loads a full block before compute begins.
                  // Requires: in0_cb >= Kt pages, in1_cb >= Kt*Nt pages.
    NoWait        // Caller manages all CB synchronization; cb_wait_front and cb_pop_front
                  // are skipped entirely. Use when the caller guarantees tiles are available.
};

}  // namespace matmul_tile_config

/**
 * matmul_tile: tiled matrix multiplication C = A × B using the LLK matmul_tiles API.
 *
 * Wraps mm_init + matmul_tiles to perform tile-by-tile matrix multiplication.
 * Reads A tiles from in0_cb (MK tile layout: [batch, Mt, Kt]) and B tiles from in1_cb
 * (KN tile layout: [batch, Kt, Nt]), and writes C tiles to out_cb (MN tile layout:
 * [batch, Mt, Nt]).
 *
 * The inner loop order is batch × Mt × Nt × Kt. This must match the CB production order
 * from the reader kernel.
 *
 * Supports optional batch dimension: each batch slice is an independent [Mt×Kt] × [Kt×Nt]
 * matmul.
 *
 * PREREQUISITE: compute_kernel_hw_startup(in0_cb, in1_cb, out_cb) must be called at the
 * start of the compute kernel before using this function. Use the three-argument form because
 * srcA (in0_cb) and srcB (in1_cb) are different circular buffers.
 *
 * ── Template Parameters (compile-time) ──────────────────────────────────────
 *
 *   in0_cb          — Input CB index for matrix A (0–31, tiled data, MK layout).
 *   in1_cb          — Input CB index for matrix B (0–31, tiled data, KN layout).
 *   out_cb          — Output CB index for matrix C (0–31, tiled data, MN layout).
 *                     Must differ from both in0_cb and in1_cb (enforced by static_assert).
 *   init_uninit_mode — Init/uninit lifecycle control (default: InitAndUninit).
 *                     UninitOnly and Neither are both no-ops since there is no mm_uninit
 *                     in the LLK API. Included for API symmetry with tilize/untilize.
 *   wait_mode       — Input synchronization strategy (default: WaitPerTile).
 *   reconfig_mode   — Register datatype reconfiguration (default: UnpackAndPackReconfigure).
 *                     Applied before mm_init. Use when the previous op used a different
 *                     data format to avoid stale format configuration.
 *
 * ── Runtime Parameters ───────────────────────────────────────────────────────
 *
 *   Mt    — Number of output tile rows (= A tile rows). Must be > 0.
 *   Nt    — Number of output tile columns (= B tile columns). Must be > 0.
 *   Kt    — Number of inner-dimension tiles (= A tile cols = B tile rows). Must be > 0.
 *   batch — Number of independent matmul batch slices (default: 1). Must be > 0.
 *
 * ── CB Sizing Requirements ───────────────────────────────────────────────────
 *
 *   WaitPerTile (default): in0_cb >= 1 page, in1_cb >= 1 page, out_cb >= 1 page.
 *   WaitUpfront: in0_cb >= Kt pages, in1_cb >= Kt*Nt pages, out_cb >= 1 page.
 *   All CBs use tiled data format (not row-major).
 *
 * ── Examples ─────────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/matmul_tile_helpers.hpp"
 *
 *   // Hardware init — must come first; three-argument form required for matmul
 *   // because srcA and srcB are different CBs.
 *   compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
 *
 *   // 1. Basic usage (all defaults — standalone matmul)
 *   compute_kernel_lib::matmul_tile<cb_in0, cb_in1, cb_out>(Mt, Nt, Kt);
 *
 *   // 2. Register reconfiguration when transitioning from a different op mode
 *   //    (e.g., after an eltwise op that used different data formats)
 *   using namespace compute_kernel_lib::matmul_tile_config;
 *   compute_kernel_lib::matmul_tile<cb_in0, cb_in1, cb_out,
 *       InitUninitMode::InitAndUninit,
 *       WaitMode::WaitPerTile,
 *       ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(Mt, Nt, Kt);
 *
 *   // 3. WaitUpfront mode — tiles pre-loaded by reader before compute starts;
 *   //    CBs must be sized: in0_cb >= Kt pages, in1_cb >= Kt*Nt pages.
 *   compute_kernel_lib::matmul_tile<cb_in0, cb_in1, cb_out,
 *       matmul_tile_config::InitUninitMode::InitAndUninit,
 *       matmul_tile_config::WaitMode::WaitUpfront>(Mt, Nt, Kt);
 *
 *   // 4. NoWait mode — caller guarantees all tiles are in CBs before this call
 *   compute_kernel_lib::matmul_tile<cb_in0, cb_in1, cb_out,
 *       matmul_tile_config::InitUninitMode::InitAndUninit,
 *       matmul_tile_config::WaitMode::NoWait>(Mt, Nt, Kt);
 *
 *   // 5. Back-to-back matmul calls — skip redundant init between them using
 *   //    InitOnly / Neither / UninitOnly (UninitOnly is a no-op for matmul)
 *   compute_kernel_lib::matmul_tile<cb_a0, cb_b0, cb_c0,
 *       matmul_tile_config::InitUninitMode::InitOnly>(Mt, Nt, Kt);    // first: init
 *   compute_kernel_lib::matmul_tile<cb_a1, cb_b1, cb_c1,
 *       matmul_tile_config::InitUninitMode::Neither>(Mt, Nt, Kt);     // middle: no init
 *   compute_kernel_lib::matmul_tile<cb_a2, cb_b2, cb_c2,
 *       matmul_tile_config::InitUninitMode::UninitOnly>(Mt, Nt, Kt);  // last: no-op uninit
 */
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    matmul_tile_config::InitUninitMode init_uninit_mode = matmul_tile_config::InitUninitMode::InitAndUninit,
    matmul_tile_config::WaitMode wait_mode = matmul_tile_config::WaitMode::WaitPerTile,
    matmul_tile_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        matmul_tile_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
    bool transpose = false>
ALWI void matmul_tile(uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t batch = 1);

}  // namespace compute_kernel_lib

#include "matmul_tile_helpers.inl"
