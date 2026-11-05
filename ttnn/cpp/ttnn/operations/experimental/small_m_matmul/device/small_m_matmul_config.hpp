// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/experimental/small_m_matmul/device/small_m_matmul_plan.hpp"

namespace ttnn::experimental::prim {

// Public manual-config knobs for the small-M DRAM-BW-optimal matmul. Field names mirror the pure
// host planner's plan::SmallMConfig. All values are in tiles / slice counts.
//
// IMPORTANT (program-cache identity): every field here feeds compile-time kernel args, so this whole
// struct must be part of SmallMMatmulParams (the device-op operation_attributes) and is hashed via
// the framework's default reflection-based program hash (same mechanism as MinimalMatmulConfig).
// Manual execution config. The recommended production path is `config=std::nullopt` (auto-pick via
// auto_select_config); an explicit config is for reproducibility / tuning. The Python binding has NO
// zero-argument constructor (all five fields must be set explicitly). In C++ the struct keeps aggregate
// member defaults because the factory/planner normalise 0 -> 1 (`cfg.k_slices ? : 1`), so a
// value-initialised config is the DEGENERATE all-ones single-band layout: Pk=Ns=Sm=kb=1, nsb=0 => exactly
// 8 worker cores. That is valid-but-slow, not incorrect — and it is NOT silently accepted where it would not
// fit: the planner (build_plan) rejects it with an explicit "L1 over budget" error on deep-K shapes and runs
// it (using only 8 cores) on small ones. No extra C++ guard is added: removing the defaults cannot prevent
// the all-ones config (0 normalises back to 1), and a required-args constructor would break the picker's
// designated-initialiser construction — the planner's feasibility check is the backstop.
struct SmallMMatmulConfig {
    uint32_t k_slices{1};          // Pk : split-K depth (>=1). Reduction only when >1.
    uint32_t n_slices{1};          // Ns : N-slices per bank-band.
    uint32_t m_slices{1};          // Sm : M-split factor.
    uint32_t k_block_tiles{1};     // kb : K-block depth fed to compute (tiles).
    uint32_t n_subblock_tiles{0};  // nsb: N-subblock width (tiles). 0 => full N_own.
};

namespace plan = ttnn::operations::experimental::small_m_matmul::plan;

// Auto-select a (Pk, Ns, Sm, kb, nsb) config for a shape given in TILE counts. Ported from the
// validated FLUX/LTX picker (measured-table oracle + cost-model fallback): a lookup
// table for the production shapes, else enumerate feasible candidates and pick the min-cost one
// (Sm=1 anchor + narrow-N M-split hysteresis). Used when the caller passes config=None.
// `fusion` describes which fused-epilogue operands will be present. It matters because those operands are
// extra L1 (c_4..c_6, plus c_10 under reduce-scatter), so a config that is feasible unfused can overflow L1
// when fused. Defaults to unfused.
// `num_banks` is the device's DRAM bank count G (== in0 ring size; 8 on P150, 7 on P100). The measured lookup
// table is 8-bank-only; other G use the (G-parameterised) cost model.
SmallMMatmulConfig auto_select_config(
    uint32_t Mt, uint32_t Kt, uint32_t Nt, uint32_t num_banks, const plan::FusionInputs& fusion = {});

// Number of DRAM banks the op width-shards in1 across (== device->num_dram_channels(), validated >= 2).
uint32_t num_dram_banks(const tt::tt_metal::IDevice* device);

// Bank-adjacent logical worker per DRAM bank (index == bank id) for the chip at `coord`, via the per-device
// MeshDevice::get_optimal_dram_bank_to_logical_worker_assignment(noc, coord) overload.
std::vector<plan::PlanXY> optimal_bank_workers(
    tt::tt_metal::distributed::MeshDevice* device,
    const tt::tt_metal::distributed::MeshCoordinate& coord,
    tt::tt_metal::NOC noc);

// Device adapter: read (Mt, Kt, Nt) from the tensors, fetch the compute grid + the bank-adjacent worker
// assignment of the chip at `coord` (per-device: harvesting differs chip to chip), translate SmallMMatmulConfig
// -> plan::SmallMConfig, and run the pure planner. When cfg is nullopt, auto_select_config picks the config.
// Returns the plan result verbatim; the caller must TT_FATAL on !ok() with plan.error.
plan::PlanResult make_and_build_plan(
    tt::tt_metal::distributed::MeshDevice* device,
    const tt::tt_metal::distributed::MeshCoordinate& coord,
    const Tensor& in0,
    const Tensor& in1,
    const std::optional<SmallMMatmulConfig>& cfg,
    const plan::FusionInputs& fusion = {});

// Build the canonical DRAM width-sharded MemoryConfig for the small-M in1 (weight) tensor.
//
// Shard layout: [Kt rows, N_padded / G] tiles across all G DRAM banks (WIDTH_SHARDED, ROW_MAJOR), where G is
// the device's DRAM bank count (8 on P150, 7 on P100). Padding is CONFIG-INDEPENDENT: K is not padded beyond
// tile alignment and N is rounded up to a multiple of G tiles (so the shard spec is a function of (K, N, G)
// only, never of Pk/kb/Ns/nsb). A weight laid out for one bank count is rejected on a device with another.
//
// weight_shape: logical [.., K, N] (elements). dtype/device carried for API completeness + validation.
tt::tt_metal::MemoryConfig create_small_m_weight_memory_config(
    const ttnn::Shape& weight_shape, tt::tt_metal::DataType dtype, tt::tt_metal::IDevice* device);

}  // namespace ttnn::experimental::prim
