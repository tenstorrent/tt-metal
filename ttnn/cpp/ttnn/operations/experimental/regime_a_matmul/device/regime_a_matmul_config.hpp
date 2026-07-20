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

// RegimeADiag (the test-only diagnostic ablation bitmask) is INTERNAL-ONLY and deliberately NOT declared here
// — this header is reachable from the public op header (regime_a_matmul.hpp) and the nanobind TU, so the enum
// lives in the internal-only device/regime_a_matmul_diag.hpp, included only by the program factory. The
// diag_mask field in RegimeAMatmulParams is a plain uint32_t, so nothing here needs the enum.

namespace plan = ttnn::operations::experimental::regime_a_matmul::plan;

// Auto-select a (Pk, Ns, Sm, kb, nsb) config for a shape given in TILE counts. Ported from the
// validated FLUX/LTX picker (tools/mm_sweep/picker_table.py oracle + cost-model fallback): a lookup
// table for the production shapes, else enumerate feasible candidates and pick the min-cost one
// (Sm=1 anchor + narrow-N M-split hysteresis). Used when the caller passes config=None.
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
