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

// Q tile rows the compute kernel processes per chunk. STANDARD pairs Q rows, so its odd chunks are padded with
// one zero row (read as zeros, output dropped) and its kernel compiles a single group path. LOW_PRECISION
// computes an odd chunk as is, ending with a single-row group: odd chunks let it balance Q chunks over the
// grid (e.g. Wan2.2 720p ring: Q288, 330 chunks on 110 cores).
uint32_t recipe_compute_q_tiles(const PrecisionPolicy& policy, uint32_t q_tiles);

// Fused STANDARD / LOW_PRECISION chunks add CBs 29-31 (recipe_compute_program). A layout whose fused CBs do not
// fit L1 drops them and the SDPA_RECIPE_FUSED define, so the kernel runs every K chunk on the reduce path.
// Returns the bytes freed (0 when the layout was not fused).
uint64_t recipe_drop_fused(tt::tt_metal::ProgramDescriptor::CBDescriptors& cbs, tt::tt_metal::KernelDescriptor::Defines& defines);
uint64_t recipe_drop_fused(tt::tt_metal::ProgramDescriptor::CBDescriptors& cbs, std::map<std::string, std::string>& defines);

// Recipe QK/PV matmul subblock width for a K chunk or head dim of `tiles` tiles: the largest of 4, 2 and 1
// dividing it (SDPA_RECIPE_QK_W / SDPA_RECIPE_PV_W). Shared by the dense, ring and exp ring recipe hosts.
uint32_t recipe_subblock_width(uint32_t tiles);

tt::tt_metal::ProgramDescriptor recipe_compute_program(
    const PrecisionPolicy& policy,
    const CoreRangeSet& grid,
    uint32_t k_chunks,
    uint32_t q_tiles = 8,
    uint32_t k_tiles = 16,
    uint32_t d_tiles = 4);

PrecisionPolicy resolve_recipe_policy(
    const Tensor& q,
    const Tensor& k,
    ttnn::transformer::SDPAPrecision precision,
    std::optional<float> scale,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<SDPAProgramConfig>& program_config);

std::tuple<Tensor, Tensor> run_joint_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& joint_q,
    const Tensor& joint_k,
    const Tensor& joint_v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config);

// Checks an additive attn_mask for the dense recipe path (shape, dtype, layout) before any dispatch.
void validate_recipe_mask(const Tensor& q, const Tensor& k, const Tensor& mask, const PrecisionPolicy& policy);

// attn_mask: optional additive mask [1|B, 1|H, Sq, Sk] (BF16/BFP8/BFP4, tiled DRAM), already
// multiplied by 1/scale like legacy SDPA; it is L1-accumulated onto the QK scores before the row max.
Tensor run_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config,
    const std::optional<Tensor>& attn_mask = std::nullopt);

}  // namespace ttnn::operations::transformer::sdpa::detail
