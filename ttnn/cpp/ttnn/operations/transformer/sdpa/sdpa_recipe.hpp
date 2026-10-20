// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "sdpa_precision_policy.hpp"
#include "sdpa_recipe_blocking.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include "sdpa.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::transformer::sdpa::detail {

RecipeSelection select_recipe(ttnn::transformer::SDPAPrecision precision, DataType kv_type);
DataType recipe_kv_dtype(KVStorage storage);
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

// Total circular-buffer bytes of a program.
uint64_t recipe_cb_bytes(const tt::tt_metal::ProgramDescriptor& program);

// CB budget below the lowest live L1 buffer: global semaphores and persistent buffers occupy the top of L1 in a
// pipeline, and static CBs must not overlap them.
uint64_t recipe_free_l1(tt::tt_metal::IDevice& device);

// Fused STANDARD / FAST chunks add CBs 29-31 (recipe_compute_program).
bool recipe_fused_cb(const tt::tt_metal::CBDescriptor& cb);

// A layout whose fused CBs do not fit L1 drops them and the SDPA_RECIPE_FUSED define, so the kernel runs every K
// chunk on the reduce path. An odd STANDARD Q chunk keeps them (its unfused kernel cannot build the single-row group;
// recipe_compute_q_tiles). `defines` is a KernelDescriptor::Defines or a std::map. Returns the bytes freed (0 when
// nothing was dropped).
template <typename Defines>
uint64_t recipe_drop_fused(tt::tt_metal::ProgramDescriptor::CBDescriptors& cbs, Defines& defines, uint32_t q_tiles) {
    auto named = [](const char* name) { return [name](const auto& define) { return define.first == name; }; };
    if (q_tiles % 2 != 0 && std::ranges::none_of(defines, named("SDPA_RECIPE_LOFI"))) {
        return 0;
    }
    uint64_t freed = 0;
    for (const auto& cb : cbs) {
        freed += recipe_fused_cb(cb) ? cb.total_size : 0;
    }
    cbs.erase(std::remove_if(cbs.begin(), cbs.end(), recipe_fused_cb), cbs.end());
    std::erase_if(defines, named("SDPA_RECIPE_FUSED"));
    return freed;
}

// Recipe QK/PV matmul subblock width for a K chunk or head dim of `tiles` tiles: the largest of 4, 2 and 1
// dividing it (SDPA_RECIPE_QK_W / SDPA_RECIPE_PV_W). Shared by the dense, ring and exp ring recipe hosts.
uint32_t recipe_subblock_width(uint32_t tiles);

tt::tt_metal::ProgramDescriptor recipe_compute_program(
    const PrecisionPolicy& policy,
    const CoreRangeSet& grid,
    uint32_t k_chunks,
    uint32_t q_tiles,
    uint32_t k_tiles,
    uint32_t d_tiles);

// Throws when the kernels are built without SFPLOADMACRO (TT_METAL_DISABLE_SFPLOADMACRO, set on the simulator): the
// recipes' exps record SFPLOADMACRO programs and compute wrong results without them. resolve_recipe_policy and
// validate_ring_recipe_inputs call it, so every recipe call checks it before dispatch.
void require_recipe_sfploadmacro();

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

// Ring and exp ring device operations: the recipe checks both ops share. Blackhole; BF16 Q and joint Q; K, V, the
// gathered K/V buffers and the joint K/V share one BF16, BFP8 or BFP4 type; supported geometry; default scale.
void validate_ring_recipe_inputs(
    RecipeOp op,
    ttnn::transformer::SDPAPrecision precision,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& gathered_k,
    const Tensor& gathered_v,
    const std::optional<Tensor>& joint_q,
    const std::optional<Tensor>& joint_k,
    const std::optional<Tensor>& joint_v,
    uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    std::optional<float> scale);

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
