// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sdpa_precision_policy.hpp"
#include "sdpa_recipe_blocking.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt_stl/reflection.hpp>

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
// zero row (read as zeros, output dropped). Key-range calls (`keyed`) pad FAST's odd chunk too: its single-row
// group mishandles a row whose keys are all masked in the Q chunk's first K chunk.
uint32_t recipe_compute_q_tiles(
    const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, bool masked = false, bool keyed = false);

// Total circular-buffer bytes of a program.
uint64_t recipe_cb_bytes(const tt::tt_metal::ProgramDescriptor& program);

// CB budget below the lowest live L1 buffer: global semaphores and persistent buffers occupy the top of L1 in a
// pipeline, and static CBs must not overlap them.
uint64_t recipe_free_l1(tt::tt_metal::IDevice& device);

// A caller's grid must have cores (the ops' common argument rule).
void validate_recipe_grid(const std::optional<SDPAProgramConfig>& program_config);

// What a recipe call resolves before the program-cache lookup (both are part of the program hash): the blocking the
// op chooses and the L1 its layout may use.
struct RecipeResolution {
    std::optional<SDPAProgramConfig> program_config;
    uint64_t l1_budget = 0;
};

// The memo key of a call's resolution: everything it reads (the caller's attributes and the input tensor specs, as
// the program hash sees them) plus the device and its free L1.
template <typename... Inputs>
std::size_t recipe_resolution_key(const Tensor& q, const Inputs&... inputs) {
    return ttsl::hash::hash_objects_with_default_seed(q.device()->id(), recipe_free_l1(*q.device()), inputs...);
}

// The resolution for `key`, from `resolve` the first time the key is seen (a repeated call does no planning). Only
// successful resolutions are kept; the memo is bounded.
RecipeResolution memoized_recipe_resolution(std::size_t key, const std::function<RecipeResolution()>& resolve);

// Fused STANDARD / FAST chunks add CBs 29-31 (recipe_compute_program).
bool recipe_fused_cb(const tt::tt_metal::CBDescriptor& cb);

// The recipe compute kernels select their behavior through named compile-time arguments (kernel side:
// device/kernels/compute/streaming/recipe_config.hpp). recipe_compute_program sets every one of them on its compute
// kernel; the ring and exp ring factories copy them (the "recipe_" names) onto their kernels.
using NamedCompileTimeArgs = tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs;
// Sets `name` to `value`, adding the argument if absent.
void set_named_arg(NamedCompileTimeArgs& args, std::string_view name, uint32_t value);
// The value of `name`, which must be present.
uint32_t named_arg(const NamedCompileTimeArgs& args, std::string_view name);
// The recipe arguments ("recipe_" names) of a recipe compute kernel's named compile-time arguments.
NamedCompileTimeArgs recipe_named_args(const NamedCompileTimeArgs& args);

// A layout whose fused CBs do not fit L1 drops them and clears recipe_fused in the recipe compute kernel's named
// arguments `args`, so the kernel runs every K chunk on the reduce path. An odd STANDARD Q chunk keeps them (its
// unfused kernel cannot build the single-row group; recipe_compute_q_tiles). Returns the bytes freed (0 when nothing
// was dropped).
uint64_t recipe_drop_fused(
    tt::tt_metal::ProgramDescriptor::CBDescriptors& cbs, NamedCompileTimeArgs& args, uint32_t q_tiles);

// Recipe QK/PV matmul subblock width for a K chunk or head dim of `tiles` tiles: the largest of 4, 2 and 1
// dividing it (recipe_qk_w / recipe_pv_w). Shared by the dense, ring and exp ring recipe hosts.
uint32_t recipe_subblock_width(uint32_t tiles);

// The kernel optimization level a recipe factory sets: `level`, or the default when the watcher is enabled. Watcher
// builds are -Os (the recipe kernels' WATCHER_ENABLED pragma), and their link must stay at the default level: at -O2
// the LLK's address-mode setup is not inlined into the -Os code and its asm constants do not resolve.
std::optional<tt::tt_metal::KernelBuildOptLevel> recipe_opt_level(tt::tt_metal::KernelBuildOptLevel level);

// `scale` is the softmax scale the exp folds in (default 1/sqrt(head dim)). The ring and exp ring factories pass
// theirs to the compute kernel themselves and only take the CB layout, the recipe_ named arguments, defines and
// compute config from here.
// `vd_tiles`: V / output head dim in tiles when it differs from the QK head dim `d_tiles` (MLA; 0: the same).
// `v_storage`: V's storage when it differs from K's (the policy's); the unpacker expands either into the same BF16
// source registers, so only CB 2's format changes.
// `q_format`: Q's format (BF16, BFP8 or BFP4): CB 0 holds Q as stored, and the output CB 16 is packed in it (the
// output keeps Q's type). The unpacker expands Q into the same BF16 source registers.
tt::tt_metal::ProgramDescriptor recipe_compute_program(
    const PrecisionPolicy& policy,
    const CoreRangeSet& grid,
    uint32_t k_chunks,
    uint32_t q_tiles,
    uint32_t k_tiles,
    uint32_t d_tiles,
    std::optional<float> scale = std::nullopt,
    uint32_t vd_tiles = 0,
    std::optional<KVStorage> v_storage = std::nullopt,
    tt::DataFormat q_format = tt::DataFormat::Float16_b);

// K/V storage of a K/V tensor dtype (BF16, BFP8, BFP4; anything else is rejected by the recipe's validation).
KVStorage recipe_kv_storage(DataType dtype);

// Throws when the kernels are built without SFPLOADMACRO (TT_METAL_DISABLE_SFPLOADMACRO, set on the simulator): the
// recipes' exps record SFPLOADMACRO programs and compute wrong results without them. resolve_recipe_policy and
// validate_ring_recipe_inputs call it, so every recipe call checks it before dispatch.
void require_recipe_sfploadmacro();

// The recipe owns the numerics: the prims do not take a compute_kernel_config, and program_config.exp_approx_mode
// is accepted and ignored. `scale` may be any finite positive value.
PrecisionPolicy resolve_recipe_policy(
    const Tensor& q,
    const Tensor& k,
    ttnn::transformer::SDPAPrecision precision,
    std::optional<float> scale,
    const std::optional<SDPAProgramConfig>& program_config);

// The recipe's matmul fidelity (perf models); unresolved precision means STANDARD.
tt::tt_metal::MathFidelity recipe_math_fidelity(
    std::optional<ttnn::transformer::SDPAPrecision> precision, DataType kv_type);

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
    // Ring-distributed SDPA: Q holds the whole sequence and the call computes two slabs of q_slab_rows rows (whole Q
    // chunks), the output stacking them. Each device's program gets its slabs' first rows (set_recipe_q_slabs); slab
    // row r of slab s sits at global position q_offset + start[s] + r.
    uint32_t q_slab_rows = 0;  // 0: Q is computed whole
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

// The recipe program for `segments` (Q/K/V, then the joint Q/K/V; V may be K itself for MLA) writing `outputs`;
// validates the call. Called from the SDPA, JointSDPA and RingDistributedSDPA program factories on a program-cache
// miss. Buffer addresses are common runtime args bound to their buffers (kernels/recipe_args.hpp), so a cache hit
// patches them without rebuilding the descriptor.
// attn_mask: optional additive mask [1|B, 1|H, Sq, Sk] (BF16/BFP8/BFP4/FP32, tiled, interleaved), already multiplied
// by 1/scale; it is L1-accumulated onto the QK scores before the row max. The outputs are BF16, interleaved.
// `l1_budget`: the L1 the layout may use (recipe_l1_budget), hashed by the caller so a program does not depend on
// what else is live in L1.
tt::tt_metal::ProgramDescriptor recipe_program(
    const std::vector<std::array<Tensor, 3>>& segments,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config,
    const std::optional<Tensor>& attn_mask,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const RecipeKeyRange& key_range,
    const RecipeDenseOptions& options,
    const std::vector<Tensor>& outputs,
    uint64_t l1_budget);

// The L1 bytes a recipe layout may use: the free L1 below the live L1 buffers less the outputs' share, capped at the
// blocking chooser's preferred layout, so calls with room to spare share one program.
// Whether the call's explicit chunks fit the free L1 in their smallest layout (recipe_l1_bytes' minimum).
bool recipe_chunks_fit(
    const std::vector<std::array<Tensor, 3>>& segments,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config,
    const std::optional<Tensor>& attn_mask,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const RecipeKeyRange& key_range,
    const RecipeDenseOptions& options);

uint64_t recipe_l1_budget(
    const std::vector<std::array<Tensor, 3>>& segments,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config,
    const std::optional<Tensor>& attn_mask,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const RecipeKeyRange& key_range,
    const RecipeDenseOptions& options);

// One output per segment: BF16, V's head dim, the two Q slabs' rows (ring-distributed), [B, 1, Sq, H * Dv] with
// concatenated heads.
tt::tt_metal::TensorSpec recipe_output_spec(
    const Tensor& q,
    uint32_t head_dim_v,
    uint32_t q_slab_rows,
    bool output_concat_heads,
    const tt::tt_metal::MemoryConfig& output_memory_config);

// Ring-distributed SDPA: writes one device's two Q slab starts (rows of the whole sequence, whole Q chunks of
// q_chunk_rows) into a recipe program built with RecipeKeyRange::q_slab_rows.
void set_recipe_q_slabs(
    tt::tt_metal::ProgramDescriptor& program, uint32_t q_chunk_rows, const std::array<uint32_t, 2>& starts);

}  // namespace ttnn::operations::transformer::sdpa::detail
