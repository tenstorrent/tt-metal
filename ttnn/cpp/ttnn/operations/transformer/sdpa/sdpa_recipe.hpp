// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>
#include <string>
#include <vector>

#include "sdpa_precision_policy.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include "sdpa.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::transformer::sdpa::detail {

RecipeSelection select_recipe(ttnn::transformer::SDPAPrecision precision, DataType kv_type);
// Dense/joint recipes: any tile-aligned Q chunk up to 1024 rows and any tile-aligned K chunk (ring and exp ring
// validate through recipe_geometry_rejection in sdpa_recipe_blocking.hpp; L1 fit is checked when the program is
// built).
uint32_t recipe_dense_q_tiles(const std::optional<SDPAProgramConfig>& program_config);
uint32_t recipe_dense_k_tiles(const std::optional<SDPAProgramConfig>& program_config);

// Q tile rows the compute kernel processes per chunk. The fused STANDARD / FAST kernels compute an odd
// chunk as is, ending with a single-row group, which lets the chooser balance Q chunks over the grid (Wan2.2
// 720p ring: Q288, 330 chunks on 110 cores). STANDARD's unfused kernel (one-tile-wide QK subblocks, attn_mask)
// does not fit the kernel config buffer with that second group path, so there an odd chunk is padded with one
// zero row (read as zeros, output dropped).
uint32_t recipe_compute_q_tiles(const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, bool masked = false);

// Fused STANDARD / FAST chunks add CBs 29-31 (recipe_compute_program). A layout whose fused CBs do not
// fit L1 drops them and the SDPA_RECIPE_FUSED define, so the kernel runs every K chunk on the reduce path.
// An odd STANDARD Q chunk keeps them (its unfused kernel cannot build the single-row group; recipe_compute_q_tiles).
// Returns the bytes freed (0 when nothing was dropped).
uint64_t recipe_drop_fused(
    tt::tt_metal::ProgramDescriptor::CBDescriptors& cbs, tt::tt_metal::KernelDescriptor::Defines& defines, uint32_t q_tiles);
uint64_t recipe_drop_fused(
    tt::tt_metal::ProgramDescriptor::CBDescriptors& cbs, std::map<std::string, std::string>& defines, uint32_t q_tiles);

// Recipe QK/PV matmul subblock width for a K chunk or head dim of `tiles` tiles: the largest of 4, 2 and 1
// dividing it (SDPA_RECIPE_QK_W / SDPA_RECIPE_PV_W). Shared by the dense, ring and exp ring recipe hosts.
uint32_t recipe_subblock_width(uint32_t tiles);

// `scale` is the softmax scale the exp folds in (default 1/sqrt(head dim)). The ring and exp ring factories pass
// theirs to the compute kernel themselves and only take the CB layout, defines and compute config from here.
tt::tt_metal::ProgramDescriptor recipe_compute_program(
    const PrecisionPolicy& policy,
    const CoreRangeSet& grid,
    uint32_t k_chunks,
    uint32_t q_tiles = 8,
    uint32_t k_tiles = 16,
    uint32_t d_tiles = 4,
    std::optional<float> scale = std::nullopt);

// The recipe owns the numerics: compute_kernel_config (fidelity, approx mode, FP32 dest, L1 accumulation) and
// program_config.exp_approx_mode are accepted and ignored. `scale` may be any finite positive value.
PrecisionPolicy resolve_recipe_policy(
    const Tensor& q,
    const Tensor& k,
    ttnn::transformer::SDPAPrecision precision,
    std::optional<float> scale,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<SDPAProgramConfig>& program_config);

// Q (and joint Q) must already be BF16 (recipe_bf16_query); K/V are BF16, BFP8 or BFP4 per the policy.
std::tuple<Tensor, Tensor> run_joint_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& joint_q,
    const Tensor& joint_k,
    const Tensor& joint_v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config,
    std::optional<float> scale = std::nullopt);

// The recipe kernels read Q as BF16. A BFP8/BFP4 Q (legacy SDPA accepts them) is widened to BF16 in DRAM first;
// the conversion is exact. BF16 Q is returned as is.
Tensor recipe_bf16_query(const Tensor& q);

// L1 bytes per core an interleaved output of `pages` tiles of `page_bytes` takes in `memory_config` (0 for DRAM).
// The dense blocking chooser reserves them, since the output is allocated after it runs.
uint64_t recipe_output_l1_bytes(
    const Tensor& q, uint64_t pages, uint32_t page_bytes, const tt::tt_metal::MemoryConfig& memory_config);

// Checks an additive attn_mask for the dense recipe path (shape, dtype, layout) before any dispatch.
void validate_recipe_mask(const Tensor& q, const Tensor& k, const Tensor& mask, const PrecisionPolicy& policy);

// attn_mask: optional additive mask [1|B, 1|H, Sq, Sk] (BF16/BFP8/BFP4, tiled, interleaved), already
// multiplied by 1/scale like legacy SDPA; it is L1-accumulated onto the QK scores before the row max.
// The BF16 output is allocated in `output_memory_config` (interleaved DRAM or L1).
Tensor run_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config,
    const std::optional<Tensor>& attn_mask = std::nullopt,
    std::optional<float> scale = std::nullopt,
    const tt::tt_metal::MemoryConfig& output_memory_config = ttnn::DRAM_MEMORY_CONFIG);

}  // namespace ttnn::operations::transformer::sdpa::detail
