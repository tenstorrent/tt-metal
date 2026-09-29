// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "qkv_causal_conv1d_silu_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

// ============================================================================
//  Tiled path of qkv_causal_conv1d_silu (TILE-layout input)
// ============================================================================
//
// The reader takes TILE input straight from the in-projection matmul. It forms the causal shift
// S_k (k = 1..3) in L1 from the current tile and a 3-row halo of the previous tile-row (or of the
// history). Compute keeps the per-tile FPU/SFPU sequence of the ROW_MAJOR path, so the result is
// bit-identical, but it runs B tiles per dest acquire and runs the inits once per tap.
//
// Work unit = step = (column block of B tiles, one tile-row mt), in block-major order:
//   step = blk * Mt + mt,   blk = step / Mt,   mt = step % Mt.
// Each core gets one contiguous step range [step_start, step_start + step_count); the cores with one
// extra step are the last ones (see distribute_steps in the factory). The range splits into
// "units" at block boundaries. The reader loads the taps once per unit (faces 0 and 1 of each tap
// tile only) into one of two weight sets, so a block switch overlaps compute.
//
// Kernel interface (kernel paths are in the factory .cpp; the metal2 JIT binds every name below):
//   reader  CTAs: block_tiles, Mt, Ct,
//                 halo_offset, zeros_offset, state_offset (scratch offsets from the 64 B aligned base)
//           RTAs: step_start, step_count
//           DFBs: producer x_in, shift, weights. Scratchpad: scratch (reader-private).
//           Tensors: input, tap0..tap3; history if QKV_CONV_HAS_HISTORY; new_state if
//                    QKV_CONV_RETURN_STATE. The two defines are always set to 0 or 1, and they are the
//                    only switch for these options: the kernel must drop the tensor accessors that are
//                    not bound, which needs the preprocessor (an if constexpr cannot hide them).
//   compute CTAs: block_tiles, Mt.   RTAs: step_start, step_count
//           DFBs: consumer x_in, shift, weights; producer + consumer partial; producer out.
//   writer  CTAs: block_tiles, Mt, Qt, Kt, Vt.   RTAs: step_start, step_count
//           DFBs: consumer out.   Tensors: q, k, v.
//
// DFB slot conventions of one step (B = block_tiles, i = 0..B-1 is the tile in the column block):
//   x_in:    S_0 = the input tile X_i at slot i (B slots per step). The DFB is a ring of whole
//            steps; the reader takes the ring depth (>= 3 steps) from the DFB size, and step s lives
//            in chunk (s - step_start) % depth.
//   shift:   S_1 at slot i, S_2 at slot B + i, S_3 at slot 2B + i (3B slots per step, 2 steps).
//            S_k = X shifted down by k rows: S_k[r] = X[r-k] for r >= k, P[32+r-k] for r < k, where
//            P is the previous tile-row (or the history, or zeros for history=None).
//   weights: tap-major, tap t of tile i at slot t*B + i (4B slots per unit, 2 units). The reader
//            loads faces 0 and 1 of each tap tile (one 1 KB read; row 0 = the tap, rows 1-15 = the
//            tap tensor's tile padding). A ROW-broadcast unpack reads faces 0 and 1 only and the
//            multiply uses row 0 only, so faces 2 and 3 of an entry are never read.
//   Compute N1: tap 0 uses S_3 (slot 2B+i), tap 1 S_2 (B+i), tap 2 S_1 (i), tap 3 x_in (i).
// Halo (the 3 rows P[29..31] of tile i) for the first step of a unit, in scratch halo
// (B x 2 halves x 128 B, left = columns 0-15, right = columns 16-31):
//   mt0 > 0:              rows 28-31 of input tile (mt0-1): the halo base (P row 29) is at +32.
//   mt0 = 0, history:     rows 0-3 of the history tile: the halo base (history row 0) is at +0.
//   mt0 = 0, no history:  the zeros region.
// Inside a unit the reader takes P rows 29-31 straight from the previous step's x_in chunk
// (face 2/3 row 13, offsets 1440/1952). The reader refills that chunk (with the input of step
// s + depth - 1) only after the shift copies of step s, which read it, are complete, and only
// after compute has popped it.
// Scratch starts uninitialized: before first use the reader zeroes the zeros region (always) and
// the state tile (with return_conv_state).
// new_state (return_conv_state): always a DRAM-interleaved TILE [1,3,Q+K+V] tensor, whatever
//   memory_config says (memory_config applies to q/k/v only), unless the caller passes a
//   pre-allocated one (conv_state_output). The reader of the step at mt = Mt-1
//   writes rows 0-2 of each tile from x_in (rows 29-31) and the other rows from the zeroed state
//   tile in scratch, so rows 3-31 are zero.
// In-place new_state (conv_state_inplace, reader define QKV_CONV_STATE_INPLACE): new_state is the
//   history buffer. The history tiles of column block b are read only by the core that owns step
//   (b, mt = 0), and the core that owns step (b, Mt-1) can be a different core that gets there first,
//   so the write above could overwrite history before it is read. Instead the owner of (b, 0) writes
//   block b's new_state after its unit-start barrier (history landed): it reads rows 28-31 of the
//   input tiles (Mt-1, block b) into the scratch stage region together with the halo, and writes rows
//   0-2 from there. Same bytes as the write above; no cross-core ordering is needed.
//
// Design reference: qwen35_2b_handoff/plan_0925/T6/design.md sections 4.5, 6.1-6.5.
// ============================================================================

namespace ttnn::experimental::prim {

namespace qkv_causal_conv1d_silu_tiled {

inline constexpr uint32_t tap_count = 4;
// B = channel_chunk_size / 32. B = 4 keeps the L1 footprint small; B = 8 fills one bf16 dest half.
inline constexpr uint32_t default_block_tiles = 4;
inline constexpr uint32_t max_block_tiles = 8;

bool is_supported_block_tiles(uint32_t block_tiles);

// channel_chunk_size that the tiled path uses when the caller gives no program_config:
// 32 * B with B = 4, or 2 or 1 when 4 does not divide Ct = (Q+K+V) / 32.
uint32_t default_channel_chunk_size(uint32_t q_width, uint32_t k_width, uint32_t v_width);

}  // namespace qkv_causal_conv1d_silu_tiled

// One per-core L1 buffer of the tiled program.
struct QkvCausalConv1dSiluTiledBufferPlan {
    std::string name;
    std::string producer;
    std::string consumer;
    uint32_t num_entries = 0;
    uint32_t entry_size = 0;
    uint32_t bytes() const { return num_entries * entry_size; }
};

// Host-only description of the tiled program: DFB table, scratchpad layout and work split.
// The factory builds its ProgramSpec from this plan, so the plan is the source of truth for
// L1 use and core count. It needs no device.
struct QkvCausalConv1dSiluTiledPlan {
    // Geometry
    uint32_t sequence = 0;
    uint32_t q_width = 0;
    uint32_t k_width = 0;
    uint32_t v_width = 0;
    uint32_t channel_chunk_size = 0;
    uint32_t block_tiles = 0;  // B
    uint32_t Mt = 0;
    uint32_t Qt = 0;
    uint32_t Kt = 0;
    uint32_t Vt = 0;
    uint32_t Ct = 0;
    uint32_t num_blocks = 0;  // Ct / B
    uint32_t num_steps = 0;   // num_blocks * Mt
    bool has_history = false;
    bool return_conv_state = false;
    bool conv_state_inplace = false;
    uint32_t tile_size = 0;

    // L1 per core. dataflow_buffers order: x_in, shift, weights, partial, out.
    std::vector<QkvCausalConv1dSiluTiledBufferPlan> dataflow_buffers;
    // Reader-private scratchpad. Offsets are from the scratch base after 64 B alignment:
    //   halo:  B tiles x 2 column halves x 128 B (4 aligned rows; rows 28-31 of the previous
    //          tile-row, or rows 0-3 of the history)
    //   zeros: zero source for local NoC copies (the halo when history is None)
    //   state: one new_state tile under construction (rows 3-31 stay zero)
    //   stage: conv_state_inplace only, right after state (the reader derives it as state + one
    //          tile): rows 28-31 of the B input tiles of tile-row Mt-1, laid out like the halo
    uint32_t scratch_align_slack = 0;
    uint32_t scratch_halo_offset = 0;
    uint32_t scratch_halo_bytes = 0;
    uint32_t scratch_zeros_offset = 0;
    uint32_t scratch_zeros_bytes = 0;
    uint32_t scratch_state_offset = 0;
    uint32_t scratch_state_bytes = 0;
    uint32_t scratch_stage_offset = 0;  // 0 and 0 bytes unless conv_state_inplace
    uint32_t scratch_stage_bytes = 0;
    uint32_t scratch_bytes = 0;  // includes scratch_align_slack
    uint32_t dfb_bytes_per_core = 0;
    uint32_t l1_bytes_per_core = 0;  // DFBs + scratchpad

    // Work split (distribute_prep over num_steps)
    tt::tt_metal::CoreCoord grid;
    kda_factory_detail::KdaPrepWorkDist work;  // wi_start = step_start, wi_count = step_count
    uint32_t min_steps_per_core = 0;
    uint32_t max_steps_per_core = 0;
    uint32_t max_tap_loads_per_core = 0;  // units (column blocks) in the largest core range
    double balance = 0.0;                 // mean steps per core / max steps per core

    uint32_t num_cores() const { return static_cast<uint32_t>(work.cores.size()); }
    std::string to_string() const;
};

// Builds the tiled plan. TT_FATALs on geometry that the tiled path does not support.
QkvCausalConv1dSiluTiledPlan make_qkv_causal_conv1d_silu_tiled_plan(
    tt::tt_metal::CoreCoord grid,
    uint32_t sequence,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    uint32_t channel_chunk_size,
    bool has_history,
    bool return_conv_state,
    uint32_t tile_size,
    bool conv_state_inplace = false);

struct QkvCausalConv1dSiluTiledProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const QkvCausalConv1dSiluParams&, const QkvCausalConv1dSiluInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
