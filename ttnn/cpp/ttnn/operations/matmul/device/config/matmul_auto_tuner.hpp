// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::matmul::auto_tune {

// ---------------------------------------------------------------------------
// Subblock tuning
// ---------------------------------------------------------------------------
//
// Picks the largest (out_subblock_h, out_subblock_w) pair that fits in the
// DEST register file and satisfies the caller's structural constraints.
//
// DEST capacity is derived from the compute kernel config (dst_full_sync_en,
// fp32_dest_acc_en) and the output tile shape — the caller does NOT compute
// dst_reg_count themselves. This matches ttnn::get_dest_reg_count / the
// kernel-side compute_kernel_lib::DEST_AUTO_LIMIT constexpr so runtime and
// host-side tuning agree.
//
// The constraint flags map 1:1 onto the matmul factories' pre-existing
// invariants:
//   subblock_w_eq_per_core_n_required: the subblock-major writer requires
//       out_subblock_w == per_core_N (single N-subblock per row-group) OR
//       out_subblock_h == 1. Relax this by setting row_major_output=true on
//       the emitted program_config and passing false here — the row-major
//       writer path absolute-offsets every pack and has no such limit.
//   subblock_h_eq_per_core_m_required: symmetric constraint used by 1D
//       mcast_in0 sharded-output configurations.
//
// Fast-path preference reflects a Phase-3 observation: subblocks with h == 1
// hit the matmul_block helper's pack_tile_block fast path (layout identical
// to legacy subblock-major, zero per-tile LLK overhead), so when multiple
// candidates tie on volume, h == 1 or w == 1 wins. Disable to replicate the
// legacy SUBBLOCK_HW_CHOICES volume-descending order for back-compat.

struct SubblockTuneInputs {
    uint32_t per_core_M = 0;
    uint32_t per_core_N = 0;

    // Used only for DST capacity derivation — the tuner calls
    // ttnn::get_dest_reg_count internally.
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config;

    // Layout constraints — default to the relaxed (row-major) form.
    bool subblock_w_eq_per_core_n_required = false;
    bool subblock_h_eq_per_core_m_required = false;

    // Optional caps (e.g. SDPA's streaming-compute uses max_subblock_h = 2).
    std::optional<uint32_t> max_subblock_h;
    std::optional<uint32_t> max_subblock_w;

    // Output tile shape for DST capacity. Defaults to 32x32 tile.
    std::optional<std::array<uint32_t, 2>> tile_shape;

    // Prefer h==1 / w==1 shapes among ties-on-volume (matmul_block fast path).
    // Set false to match the legacy SUBBLOCK_HW_CHOICES iteration order.
    bool prefer_fast_path = true;
};

struct SubblockChoice {
    uint32_t out_subblock_h = 1;
    uint32_t out_subblock_w = 1;
};

SubblockChoice determine_largest_subblock(const SubblockTuneInputs& inputs);

// ---------------------------------------------------------------------------
// K-iteration tuning (in0_block_w)
// ---------------------------------------------------------------------------
//
// Picks the largest in0_block_w that (a) divides Kt exactly, (b) fits in the
// provided L1 budget once in0/in1 block buffers are counted, (c) does not
// exceed max_in0_block_w. Larger in0_block_w means fewer outer K-loop
// iterations — amortizes matmul_init / unpack setup across more tiles.
//
// The L1 budget formula is deliberately conservative: it sums double-buffered
// in0 and in1 block tiles plus the fixed output and (optional) interm CB
// footprint. Real L1 use on the factory may be slightly larger (reader/writer
// scratch, sync CBs) — the caller should pass a budget that already excludes
// those reserves. When in doubt, under-budget rather than over-budget.

struct InBlockWTuneInputs {
    uint32_t Kt = 0;
    uint32_t per_core_M = 0;
    uint32_t per_core_N = 0;

    uint32_t in0_single_tile_size = 0;
    uint32_t in1_single_tile_size = 0;
    uint32_t out_single_tile_size = 0;
    uint32_t interm_single_tile_size = 0;  // ignored when fuse_bias is false

    bool fuse_bias = false;

    uint32_t l1_budget_bytes = 0;
    uint32_t max_in0_block_w = UINT32_MAX;  // architectural / factory cap
    uint32_t num_buffered_blocks = 2;       // typical double-buffer
};

uint32_t determine_largest_in0_block_w(const InBlockWTuneInputs& inputs);

}  // namespace ttnn::operations::matmul::auto_tune
