// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_plan.hpp"

namespace ttnn::experimental::prim {

// Public manual-config knobs for the Regime-A DRAM-BW-optimal matmul. Field names mirror the pure
// host planner's plan::RegimeAConfig. All values are in tiles / slice counts.
//
// IMPORTANT (program-cache identity): every field here feeds compile-time kernel args, so this whole
// struct must be part of RegimeAMatmulParams (the device-op operation_attributes) and is hashed via
// the framework's default reflection-based program hash (same mechanism as MinimalMatmulConfig).
struct RegimeAMatmulConfig {
    uint32_t k_slices{1};          // Pk : split-K depth (>=1). Reduction only when >1.
    uint32_t n_slices{1};          // Ns : N-slices per bank-band.
    uint32_t m_slices{1};          // Sm : M-split factor.
    uint32_t k_block_tiles{1};     // kb : K-block depth fed to compute (tiles).
    uint32_t n_subblock_tiles{0};  // nsb: N-subblock width (tiles). 0 => full N_own.
};

// Test-only diagnostic ablation bitmask (default 0 = the normal public path). NEVER exposed through the
// Python / nanobind API; set only via the internal ttnn::prim::regime_a_matmul_diag entry point. It lives
// in RegimeAMatmulParams::diag_mask so it participates in the program-cache hash — a diagnostic program can
// never alias (or be aliased by) a normal one. Each bit maps to a DIAG_* kernel #define; mask 0 adds NO
// defines and the compile is byte-identical to the public path. Ablations are non-additive critical-path
// counterfactuals (see MT8_FINDINGS.md), not correctness modes — output is intentionally unchecked.
enum RegimeADiag : uint32_t {
    DIAG_SKIP_IN1_READ = 1u << 0,     // in1_reader: suppress in1 DRAM reads/barrier/tail-init; keep CB+M-split+sems
    DIAG_SKIP_IN0_READ = 1u << 1,     // writer: suppress step-0 in0 DRAM read/barrier; keep ptr adv/CB/ring/reduce/out
    DIAG_SKIP_IN0_FORWARD = 1u << 2,  // writer: suppress ring payload write; still signal next core (no deadlock)
    DIAG_NO_REDUCE = 1u << 3,         // force bottom-band copy path everywhere; bypass reduce credits/recv/fwd
    DIAG_LOCAL_FEED = 1u << 4,        // (reserved) purely local CB feed: no DRAM/ring/fwd/M-split/reduce
    // NOTE: the bit below is a CORRECT algorithmic VARIANT (produces the right output), not an ablation.
    DIAG_IN0_SCATTER = 1u << 5,  // writer: in0 all-gather via 1 direct-scatter round instead of G-1 serial
                                 // ring rotations. Identical cb0 layout (slot d = shard rp-d), so compute +
                                 // the in1 rotated read are unchanged; removes the serial-hop critical path.
};

namespace plan = ttnn::operations::experimental::regime_a_matmul::plan;

// Auto-select a (Pk, Ns, Sm, kb, nsb) config for a shape given in TILE counts. Ported from the
// validated FLUX/LTX picker (tools/mm_sweep/picker_table.py oracle + picker_v2.py cost-model
// fallback): a lookup table for the 20 production shapes, else enumerate feasible candidates (Sm=1)
// and pick the min-cost one. Used when the caller passes config=None.
RegimeAMatmulConfig auto_select_config(uint32_t Mt, uint32_t Kt, uint32_t Nt);

// Device adapter: read (Mt, Kt, Nt) from the tensors, fetch the compute grid + bank-adjacent worker
// assignments off the device, translate RegimeAMatmulConfig -> plan::RegimeAConfig, and run the pure
// planner. When cfg is nullopt, auto_select_config picks the config. Returns the plan result verbatim;
// the caller must TT_FATAL on !ok() with plan.error.
plan::PlanResult make_and_build_plan(
    tt::tt_metal::IDevice* device, const Tensor& in0, const Tensor& in1, const std::optional<RegimeAMatmulConfig>& cfg);

// Build the canonical DRAM width-sharded MemoryConfig for the Regime-A in1 (weight) tensor.
//
// Shard layout: [K_padded_rows, N_padded / 8] tiles across 8 DRAM banks (WIDTH_SHARDED, ROW_MAJOR).
// v1 pads with a CONFIG-INDEPENDENT alignment: K rounded up to a multiple of 8 tiles, N rounded up to
// a multiple of 8 tiles (so the shard spec is a function of (K, N) only, never of Pk/kb/Ns/nsb). See
// the design note in the .cpp — this is deliberately NOT the plan's config-padded Kt_s/Nt_s.
//
// weight_shape: logical [.., K, N] (elements). dtype/device carried for API completeness + validation.
tt::tt_metal::MemoryConfig create_regime_a_weight_memory_config(
    const ttnn::Shape& weight_shape, tt::tt_metal::DataType dtype, tt::tt_metal::IDevice* device);

}  // namespace ttnn::experimental::prim
