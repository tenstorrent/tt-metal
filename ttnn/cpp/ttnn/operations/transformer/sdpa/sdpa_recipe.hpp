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
// zero row (read as zeros, output dropped). Key-range calls (`keyed`) pad FAST's odd chunk too: its single-row
// group mishandles a row whose keys are all masked in the Q chunk's first K chunk.
uint32_t recipe_compute_q_tiles(
    const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, bool masked = false, bool keyed = false);

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
// `vd_tiles`: V / output head dim in tiles when it differs from the QK head dim `d_tiles` (MLA; 0: the same).
tt::tt_metal::ProgramDescriptor recipe_compute_program(
    const PrecisionPolicy& policy,
    const CoreRangeSet& grid,
    uint32_t k_chunks,
    uint32_t q_tiles = 8,
    uint32_t k_tiles = 16,
    uint32_t d_tiles = 4,
    std::optional<float> scale = std::nullopt,
    uint32_t vd_tiles = 0);

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

// Which keys each query row sees (the K-range + edge-mask model, dataflow/recipe_key_range.hpp). Query row q
// sits at global position q_offset + q and sees the keys k with
//   causal:                  k <= q
//   sliding_window w:        k > q - w (causal), |k - q| <= w / 2 (otherwise)
//   segments (windowed):     k in q's window [cu[i], cu[i + 1]) of cu_window_seqlens
// The reader skips K chunks no row sees, masks the edge chunks with generated {0, -2^100} tiles and leaves the
// rest unmasked, so compute keeps its one loop.
struct RecipeKeyRange {
    bool causal = false;
    uint32_t sliding_window = 0;  // 0: none
    uint32_t q_offset = 0;
    std::optional<Tensor> q_offset_tensor;  // device int32 [1]: overrides q_offset when the kernel runs (trace-safe)
    std::optional<Tensor> segments;         // cu_window_seqlens, int32/uint32 [n], row-major
    // Chunked prefill: K/V are cache blocks [blocks, KV heads, block size, D] and this int32 [B, blocks per sequence]
    // page table names each sequence's blocks in order; its K length is blocks per sequence x block size.
    std::optional<Tensor> page_table;
    // This call's view of a shared paged cache (block size, KV heads) when the cache was allocated for another
    // layer's shape; inactive: the cache's shape.
    PagedCacheGeometryOverride paged_geometry;
    bool active() const { return causal || sliding_window > 0 || segments.has_value(); }
};

// Dense recipe features outside the key range.
struct RecipeDenseOptions {
    // MLA: V and the output have this head dim (0: Q's). V may be K itself (the same tensor): its first head_dim_v
    // columns are read as V.
    uint32_t head_dim_v = 0;
    // Per-head sink logits [1, H, 1, 1] (BF16 or FP32, tiled, DRAM): each head's softmax denominator gains
    // exp(scale * sink), as legacy SDPA's attention_sink (unscaled logits).
    std::optional<Tensor> attention_sink;
    // Write the output as [B, 1, Sq, H * Dv] (heads concatenated per row) instead of [B, H, Sq, Dv].
    bool output_concat_heads = false;
};

// Logical K length of a recipe call: a paged cache's blocks per sequence x block size, else K's sequence length.
uint32_t recipe_k_rows(const Tensor& k, const RecipeKeyRange& key_range);

// L1 bytes the key-range CBs add to a recipe layout (mask row groups excluded): the control pages, the all-masked
// template tile and the scratch page for the device tensors.
uint32_t recipe_key_range_extra_bytes(const RecipeKeyRange& key_range);

// L1 bytes the dense options add to a recipe layout (the sink page CB).
uint32_t recipe_dense_options_extra_bytes(const RecipeDenseOptions& options);

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
    const tt::tt_metal::MemoryConfig& output_memory_config = ttnn::DRAM_MEMORY_CONFIG,
    const RecipeKeyRange& key_range = {},
    const RecipeDenseOptions& options = {});

}  // namespace ttnn::operations::transformer::sdpa::detail
