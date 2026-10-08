// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_recipe.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <set>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt-metalium/experimental/kernel_build_options.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/generic/generic_op.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::generic {
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor);
}  // namespace ttnn::operations::generic

namespace ttnn::operations::transformer::sdpa::detail {
using namespace tt::tt_metal;

RecipeSelection select_recipe(ttnn::transformer::SDPAPrecision precision, DataType kv_type) {
    using ttnn::transformer::SDPAPrecision;
    Recipe recipe;
    switch (precision) {
        case SDPAPrecision::STANDARD: recipe = Recipe::B; break;
        case SDPAPrecision::BALANCED: recipe = Recipe::C; break;
        case SDPAPrecision::ACCURATE: recipe = Recipe::D; break;
        case SDPAPrecision::FAST: recipe = Recipe::E; break;
        default: TT_THROW("Unknown SDPA precision recipe");
    }
    const auto storage = kv_type == DataType::BFLOAT4_B   ? KVStorage::BFP4
                         : kv_type == DataType::BFLOAT8_B ? KVStorage::BFP8
                                                          : KVStorage::BF16;
    return {recipe, storage};
}

uint32_t recipe_dense_q_tiles(const std::optional<SDPAProgramConfig>& program_config) {
    const uint32_t q_chunk = program_config ? program_config->q_chunk_size : 256;
    // Any tile-aligned Q chunk up to the recurrent-state arrays (32 tile rows); L1 fit is checked separately.
    TT_FATAL(
        q_chunk % 32 == 0 && q_chunk >= 32 && q_chunk <= 32 * 32,
        "Named SDPA recipes support tile-aligned Q chunks from 32 to 1024 rows, got {}",
        q_chunk);
    return q_chunk / 32;
}

uint32_t recipe_dense_k_tiles(const std::optional<SDPAProgramConfig>& program_config) {
    const uint32_t k_chunk = program_config ? program_config->k_chunk_size : 512;
    TT_FATAL(
        k_chunk % 32 == 0 && k_chunk >= 32, "Named SDPA recipes support tile-aligned K chunks, got {}", k_chunk);
    return k_chunk / 32;
}

uint32_t recipe_compute_q_tiles(
    const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, bool masked, bool keyed) {
    const bool unfused_standard =
        policy.selection.recipe == Recipe::B && (masked || recipe_subblock_width(k_tiles) < 2);
    const bool paired = !policy.fp32_destination;
    return (unfused_standard || (keyed && paired)) && q_tiles % 2 != 0 ? q_tiles + 1 : q_tiles;
}

namespace {
uint64_t drop_fused_cbs(ProgramDescriptor::CBDescriptors& cbs) {
    uint64_t freed = 0;
    auto fused_cb = [](const CBDescriptor& cb) {
        const uint8_t index = cb.format_descriptors.front().buffer_index;
        return index == 29 || index == 30 || index == 31;
    };
    for (const auto& cb : cbs) {
        freed += fused_cb(cb) ? cb.total_size : 0;
    }
    cbs.erase(std::remove_if(cbs.begin(), cbs.end(), fused_cb), cbs.end());
    return freed;
}
}  // namespace

uint64_t recipe_drop_fused(ProgramDescriptor::CBDescriptors& cbs, KernelDescriptor::Defines& defines, uint32_t q_tiles) {
    const bool lofi = std::any_of(defines.begin(), defines.end(), [](const auto& d) { return d.first == "SDPA_RECIPE_LOFI"; });
    if (q_tiles % 2 != 0 && !lofi) {
        return 0;
    }
    const uint64_t freed = drop_fused_cbs(cbs);
    std::erase_if(defines, [](const auto& define) { return define.first == "SDPA_RECIPE_FUSED"; });
    return freed;
}

uint64_t recipe_drop_fused(
    ProgramDescriptor::CBDescriptors& cbs, std::map<std::string, std::string>& defines, uint32_t q_tiles) {
    if (q_tiles % 2 != 0 && !defines.contains("SDPA_RECIPE_LOFI")) {
        return 0;
    }
    const uint64_t freed = drop_fused_cbs(cbs);
    defines.erase("SDPA_RECIPE_FUSED");
    return freed;
}

uint32_t recipe_subblock_width(uint32_t tiles) { return tiles % 4 == 0 ? 4 : tiles % 2 == 0 ? 2 : 1; }

namespace {
// Legacy SDPA's granularity rule: the largest value <= limit that divides the tile count.
uint32_t recipe_granularity(uint32_t tiles, uint32_t limit) {
    uint32_t g = std::min(tiles, limit);
    while (g > 1 && tiles % g != 0) {
        --g;
    }
    return g;
}
}  // namespace

ProgramDescriptor recipe_compute_program(
    const PrecisionPolicy& policy,
    const CoreRangeSet& grid,
    uint32_t k_chunks,
    uint32_t q_tiles,
    uint32_t k_tiles,
    uint32_t d_tiles,
    std::optional<float> scale,
    uint32_t vd_tiles) {
    TT_FATAL(q_tiles >= 1 && k_tiles >= 1 && d_tiles >= 1, "SDPA recipes require tile-aligned chunks and head dims");
    // V, the numerator state and the output are vd tiles wide; Q and K d tiles (MLA: vd < d).
    const uint32_t vd = vd_tiles ? vd_tiles : d_tiles;
    const bool fp32 = policy.fp32_destination;
    // STANDARD and FAST keep a reference-max state in Float32 L1 (O in CB 9, l in CB 13).
    const bool ref_max = policy.recurrent_state == RecurrentState::ReferenceMaxFP32;
    const auto state_format = fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t state_bytes = fp32 ? 4096 : 2048;
    const auto kv_type = policy.selection.kv_storage == KVStorage::BF16   ? DataType::BFLOAT16
                         : policy.selection.kv_storage == KVStorage::BFP8 ? DataType::BFLOAT8_B
                                                                          : DataType::BFLOAT4_B;
    const auto kv_format = datatype_to_dataformat_converter(kv_type);
    const uint32_t kv_bytes = kv_type == DataType::BFLOAT16 ? 2048 : kv_type == DataType::BFLOAT8_B ? 1088 : 576;
    // STANDARD and FAST (the reference-max recipes) run every K chunk after a Q chunk's first on the
    // fused chunk (recipe_fused_chunk.hpp). It needs QK subblocks at least two tiles wide: the one-wide LoFi
    // matmul reuses its other operand, which the m_ref inner step does not support.
    const bool fused = (policy.selection.recipe == Recipe::B || policy.selection.recipe == Recipe::E) &&
                       recipe_subblock_width(k_tiles) >= 2;
    ProgramDescriptor program;
    auto add_cb = [&](uint8_t index, uint32_t count, uint32_t page, tt::DataFormat format) {
        CBDescriptor cb{
            .total_size = count * page,
            .core_ranges = grid,
            .format_descriptors = {{.buffer_index = index, .data_format = format, .page_size = page}}};
        if (index == 6 && fp32) {
            cb.format_descriptors.push_back({.buffer_index = 7, .data_format = format, .page_size = page});
        }
        program.cbs.push_back(std::move(cb));
    };
    // Preserve the selected families' actual buffer depths: Q double buffered;
    // K/V one slot for FP32, two slots for BF16. No hidden geometry retuning.
    // Q-row buffers scale with the Q chunk; K/V depths and per-row state do not change.
    add_cb(0, 2 * q_tiles * d_tiles, 2048, tt::DataFormat::Float16_b);
    add_cb(1, k_tiles * d_tiles * (fp32 ? 1 : 2), kv_bytes, kv_format);
    add_cb(2, k_tiles * vd * (fp32 ? 1 : 2), kv_bytes, kv_format);
    add_cb(3, 1, 2048, tt::DataFormat::Float16_b);
    add_cb(4, 1, 2048, tt::DataFormat::Float16_b);
    add_cb(5, 1, state_bytes, state_format);
    add_cb(6, q_tiles * k_tiles, state_bytes, state_format);
    if (ref_max) {
        // CB 8: one BF16 tile, the denominator rounded for the normalization matmul.
        // CB 9: per Q row, the numerator plane, then a rescaled group's chunk PV.
        // CB 12: this chunk's BF16 row sums. CB 13: the denominator.
        add_cb(8, 1, 2048, tt::DataFormat::Float16_b);
        add_cb(9, q_tiles * vd * 2, 4096, tt::DataFormat::Float32);
        add_cb(12, q_tiles, 2048, tt::DataFormat::Float16_b);
        add_cb(13, q_tiles, 4096, tt::DataFormat::Float32);
    } else {
        for (uint8_t index : {8, 9}) {
            add_cb(index, q_tiles * vd, state_bytes, state_format);
        }
        for (uint8_t index : {12, 13}) {
            add_cb(index, q_tiles, state_bytes, state_format);
        }
    }
    for (uint8_t index : {10, 11}) {
        add_cb(index, q_tiles, 2048, tt::DataFormat::Float16_b);
    }
    add_cb(14, q_tiles, state_bytes, state_format);
    if (fused) {
        // CB 31: a check unit's saturation verdict tile in the fused chunks.
        add_cb(31, 1, 2048, tt::DataFormat::Float16_b);
        // CB 30: per Q tile row, QK-subblock-width partial row sums of the fused chunks.
        add_cb(30, q_tiles * recipe_subblock_width(k_tiles), 2048, tt::DataFormat::Float16_b);
        // CB 29: QK-subblock-width copies of -e0 in K's format (the fused chunks subtract m_ref in the QK).
        add_cb(29, recipe_subblock_width(k_tiles), kv_bytes, kv_format);
    }
    add_cb(16, (fp32 ? 2 : 4) * vd, 2048, tt::DataFormat::Float16_b);
    ComputeConfigDescriptor compute_config{
        .math_fidelity = policy.pv_fidelity,
        .fp32_dest_acc_en = fp32,
        .dst_full_sync_en = false,
        .math_approx_mode = true};
    if (fp32) {
        compute_config.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        for (uint32_t cb : {5, 7, 8, 9, 12, 13, 14}) {
            compute_config.unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
        }
    }
    KernelDescriptor compute{
        .kernel_source = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa_recipe.cpp",
        .core_ranges = grid,
        .named_compile_time_args =
            {{"k_chunks", k_chunks},
             {"scale", std::bit_cast<uint32_t>(scale.value_or(1.0f / std::sqrt(static_cast<float>(d_tiles * 32))))},
             {"q_tiles", q_tiles},
             {"k_tiles", k_tiles},
             {"d_tiles", d_tiles}},
        .defines =
            {{"EXP_APPROX_MODE", "1"},
             {"STATS_GRANULARITY", std::to_string(recipe_granularity(q_tiles, fp32 ? 4 : 8))},
             {"SUB_EXP_GRANULARITY", std::to_string(recipe_granularity(k_tiles, fp32 ? 4 : 8))},
             {"MUL_BCAST_GRANULARITY", std::to_string(recipe_granularity(q_tiles * k_tiles, fp32 ? 4 : 8))},
             {"DHT_GRANULARITY", std::to_string(recipe_granularity(vd, 8))},
             {"REDUCE_GRANULARITY", std::to_string(recipe_granularity(q_tiles, fp32 ? 2 : 4))},
             {"SDPA_RECIPE_QK_W", std::to_string(recipe_subblock_width(k_tiles))},
             {"SDPA_RECIPE_PV_W", std::to_string(recipe_subblock_width(vd))}},
        .config = compute_config};
    // The recipes record their exp programs in the replay buffer once and replay them per tile from other
    // functions; the SFPI compiler's replay optimization would overwrite them (tenstorrent/tt-metal#58433).
    compute.defines.emplace_back(tt::tt_metal::experimental::DISABLE_SFPU_REPLAY_OPTIMIZATION_DEFINE, "1");
    if (fp32) {
        compute.defines.emplace_back("SDPA_RECIPE_FP32", "1");
    }
    if (policy.selection.recipe == Recipe::D) {
        compute.defines.emplace_back("SDPA_RECIPE_ACCURATE", "1");
    }
    if (policy.selection.recipe == Recipe::E) {
        compute.defines.emplace_back("SDPA_RECIPE_LOFI", "1");
    }
    if (fused) {
        compute.defines.emplace_back("SDPA_RECIPE_FUSED", "1");
    }
    if (vd != d_tiles) {
        compute.defines.emplace_back("SDPA_RECIPE_V_DHT", std::to_string(vd));
    }
    program.kernels.push_back(std::move(compute));
    return program;
}

PrecisionPolicy resolve_recipe_policy(
    const Tensor& q,
    const Tensor& k,
    ttnn::transformer::SDPAPrecision precision,
    std::optional<float> scale,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<SDPAProgramConfig>& program_config) {
    TT_FATAL(q.storage_type() == StorageType::DEVICE, "SDPA recipes require device inputs");
    // The exps take scale * (s - m) with s - m <= 0, so the scale must be positive; it is folded into the exp
    // (and the attn_mask is pre-multiplied by 1/scale), never into Q.
    TT_FATAL(
        !scale || (std::isfinite(*scale) && *scale > 0.0f),
        "SDPA recipes require a finite positive scale, got {}",
        scale.value_or(0.0f));
    TT_FATAL(
        q.device()->arch() == tt::ARCH::BLACKHOLE || q.device()->arch() == tt::ARCH::WORMHOLE_B0,
        "SDPA precision recipes support Blackhole and Wormhole B0 only");
    // The recipe owns the numerics: an explicit compute_kernel_config (math fidelity, approx mode, FP32 dest,
    // packer L1 accumulation) and exp_approx_mode are accepted and ignored, so a caller that passes a shared
    // config (or its legacy exp choice) gets the recipe it named. See SDPAPrecisionRecipes.md.
    (void)compute_kernel_config;
    (void)program_config;
    return resolve_precision_policy(select_recipe(precision, k.dtype()));
}

// L1 per core below the lowest live L1 buffer (L1 inputs, the output, global semaphores): static CBs must not
// overlap them.
static uint64_t recipe_free_l1(IDevice& device) {
    const auto lowest = device.lowest_occupied_compute_l1_address();
    const uint64_t top = lowest.has_value() ? static_cast<uint64_t>(*lowest) : device.l1_size_per_core();
    return top - device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
}

// Reject a recipe CB layout that cannot fit the device's free L1 (a fused layout first falls back to the
// unfused kernel).
static void check_recipe_l1_fit(ProgramDescriptor& program, IDevice& device, uint32_t q_chunk, uint32_t k_chunk) {
    uint64_t bytes = 0;
    for (const auto& cb : program.cbs) {
        bytes += cb.total_size;
    }
    const uint64_t available = recipe_free_l1(device);
    if (bytes > available) {
        bytes -= recipe_drop_fused(program.cbs, program.kernels.front().defines, q_chunk / 32);
    }
    TT_FATAL(
        bytes <= available,
        "SDPA recipe needs {} bytes of L1 per core at Q{}/K{}, but only {} are available; use a smaller Q or K chunk",
        bytes,
        q_chunk,
        k_chunk,
        available);
}

void validate_recipe_mask(const Tensor& q, const Tensor& k, const Tensor& mask, const PrecisionPolicy& policy) {
    const auto& qs = q.logical_shape();
    TT_FATAL(mask.storage_type() == StorageType::DEVICE, "SDPA recipe attn_mask must be on device");
    TT_FATAL(mask.device() == q.device(), "SDPA recipe attn_mask must be on the same device as Q");
    TT_FATAL(mask.layout() == Layout::TILE, "SDPA recipe attn_mask must be tilized");
    TT_FATAL(mask.tensor_spec().tile() == Tile({32, 32}), "SDPA recipe attn_mask requires 32x32 tiles");
    TT_FATAL(
        mask.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "SDPA recipe attn_mask must be interleaved (DRAM or L1)");
    TT_FATAL(
        mask.dtype() == DataType::BFLOAT16 || mask.dtype() == DataType::BFLOAT8_B ||
            mask.dtype() == DataType::BFLOAT4_B || (mask.dtype() == DataType::FLOAT32 && policy.fp32_destination),
        "SDPA recipe attn_mask must be BF16, BFP8 or BFP4 (FP32 for FP32-state recipes)");
    const auto& ms = mask.logical_shape();
    TT_FATAL(ms.rank() == 4, "SDPA recipe attn_mask must be rank four");
    TT_FATAL(
        (ms[0] == 1 || ms[0] == qs[0]) && (ms[1] == 1 || ms[1] == qs[1]) && ms[2] == qs[2] &&
            ms[3] == k.logical_shape()[2],
        "SDPA recipe attn_mask must be [1|B, 1|H, Sq, Sk], got {} for Q {} and K {}",
        ms,
        qs,
        k.logical_shape());
    const auto& mp = mask.padded_shape();
    TT_FATAL(
        mp[0] == ms[0] && mp[1] == ms[1] && mp[2] == ((ms[2] + 31) / 32) * 32 &&
            mp[3] == ((ms[3] + 31) / 32) * 32,
        "SDPA recipe attn_mask only supports minimal tile padding");
}

// CB 15 is free in the dense recipe layout (0-14 and 16 are recipe-owned; ring uses 17/18).
constexpr uint8_t kRecipeMaskCb = 15;
// Key-range calls (RecipeKeyRange): the reader's control page per Q chunk (compute: recipe_read_key_range), the
// all-masked template tile, and a scratch page for the Q offset / page-table row (64 B slots) and cu_window_seqlens.
constexpr uint8_t kRecipeKeyRangeCb = 17;
constexpr uint8_t kRecipeMaskedTileCb = 18;
constexpr uint8_t kRecipeKeyScratchCb = 19;
constexpr uint32_t kRecipeKeyRangePage = 32;
// Ring-distributed Q slabs (RecipeKeyRange::q_slab_rows): the reader's and writer's runtime args holding the two
// slabs' first Q chunks.
constexpr uint32_t kRecipeReaderSlabArg = 16;
constexpr uint32_t kRecipeWriterSlabArg = 6;
// Generated mixed mask tiles the writer reuses (causal diagonal, window edges), after the all-masked template.
constexpr uint32_t kRecipeMaskCacheTiles = 4;
// Key-range masks are BFP4 tiles: {0, -2^100} is exact there (dataflow/recipe_key_range.hpp).
constexpr uint32_t kRecipeKeyMaskPage = 576;

// The attention sink: one page per Q chunk with the head's sink logit (dataflow/reader_recipe.cpp).
constexpr uint8_t kRecipeSinkCb = 20;
constexpr uint32_t kRecipeSinkPage = 64;

namespace {
// Two halves (reader, writer) of 64 B slots for the Q offset, then cu_window_seqlens
// (dataflow/recipe_key_range.hpp: RecipeScratch); the reader's page-table row follows both halves.
uint32_t key_range_scratch_half(const RecipeKeyRange& key_range) {
    return 128 + (key_range.segments ? key_range.segments->buffer()->aligned_page_size() : 0);
}

uint32_t key_range_scratch_bytes(const RecipeKeyRange& key_range) {
    if (!key_range.q_offset_tensor && !key_range.segments && !key_range.page_table) {
        return 0;
    }
    return 2 * key_range_scratch_half(key_range) +
           (key_range.page_table ? key_range.page_table->buffer()->aligned_page_size() : 0);
}

// The paged cache's view: block size and KV heads (the geometry override, else the cache's shape).
std::pair<uint32_t, uint32_t> paged_block_view(const Tensor& k, const RecipeKeyRange& key_range) {
    const auto& geometry = key_range.paged_geometry;
    if (geometry.active()) {
        return {geometry.block_size, geometry.num_kv_heads};
    }
    return {k.logical_shape()[2], k.logical_shape()[1]};
}

// A core's share of one head's Q chunks: a contiguous [first, first + count) of the zigzag order 0, J-1, 1, J-2, ...
// (dataflow/recipe_key_range.hpp). Pairs cost the same under a causal mask; with fewer than two chunks per core the
// heaviest go alone.
std::vector<std::pair<uint32_t, uint32_t>> recipe_zigzag_split(uint32_t jobs, uint32_t chain) {
    std::vector<std::pair<uint32_t, uint32_t>> split;
    uint32_t first = 0;
    for (uint32_t r = 0; r < chain; ++r) {
        uint32_t count;
        if (jobs >= 2 * chain) {
            const uint32_t pairs = jobs / 2;
            count = 2 * (pairs / chain + (r < pairs % chain ? 1 : 0)) + (jobs % 2 != 0 && r + 1 == chain ? 1 : 0);
        } else {
            count = r < jobs - chain ? 2 : 1;
        }
        split.emplace_back(first, count);
        first += count;
    }
    TT_ASSERT(first == jobs);
    return split;
}

void validate_key_range(const Tensor& q, const RecipeKeyRange& key_range) {
    auto check_index_tensor = [&](const Tensor& t, const char* name) {
        TT_FATAL(t.storage_type() == StorageType::DEVICE && t.device() == q.device(), "{} must be on Q's device", name);
        TT_FATAL(t.dtype() == DataType::INT32 || t.dtype() == DataType::UINT32, "{} must be INT32 or UINT32", name);
        TT_FATAL(t.layout() == Layout::ROW_MAJOR, "{} must be row-major", name);
        TT_FATAL(!t.is_sharded(), "{} must be interleaved", name);
    };
    if (key_range.q_offset_tensor) {
        check_index_tensor(*key_range.q_offset_tensor, "SDPA recipe Q offset tensor");
        TT_FATAL(
            key_range.q_offset_tensor->logical_shape().volume() == 1,
            "SDPA recipe Q offset tensor must hold one value");
    }
    if (key_range.segments) {
        check_index_tensor(*key_range.segments, "cu_window_seqlens");
        const auto& shape = key_range.segments->logical_shape();
        TT_FATAL(
            shape.rank() == 1 && shape[0] >= 2 && shape[0] <= 1024,
            "cu_window_seqlens must be 1-D with 2 to 1024 entries, got {}",
            shape);
    }
    if (key_range.q_slab_rows) {
        TT_FATAL(key_range.causal, "SDPA recipe Q slabs (ring-distributed SDPA) are causal");
        TT_FATAL(!key_range.q_slab_starts.empty(), "SDPA recipe Q slabs need their start rows");
        for (const auto& [devices, starts] : key_range.q_slab_starts) {
            for (const uint32_t start : starts) {
                TT_FATAL(
                    start % 32 == 0 && start + key_range.q_slab_rows <= q.logical_shape()[2],
                    "SDPA recipe Q slab [{}, {} + {}) must be tile-aligned and inside Q's {} rows",
                    start,
                    start,
                    key_range.q_slab_rows,
                    q.logical_shape()[2]);
            }
        }
    }
    if (key_range.page_table) {
        const auto& table = *key_range.page_table;
        check_index_tensor(table, "SDPA recipe page table");
        const auto& shape = table.logical_shape();
        TT_FATAL(
            shape.rank() == 2 && shape[0] == q.logical_shape()[0] && shape[1] >= 1,
            "SDPA recipe page table must be [B, blocks per sequence], got {}",
            shape);
    } else {
        TT_FATAL(!key_range.paged_geometry.active(), "paged_cache_geometry requires a page table");
    }
}
}  // namespace

uint32_t recipe_k_rows(const Tensor& k, const RecipeKeyRange& key_range) {
    if (!key_range.page_table) {
        return k.logical_shape()[2];
    }
    return key_range.page_table->logical_shape()[1] * paged_block_view(k, key_range).first;
}

uint32_t recipe_dense_options_extra_bytes(const RecipeDenseOptions& options) {
    return options.attention_sink ? 2 * kRecipeSinkPage : 0;
}

uint32_t recipe_key_range_extra_bytes(const RecipeKeyRange& key_range) {
    return key_range.active() ? 2 * kRecipeKeyRangePage + (1 + kRecipeMaskCacheTiles) * kRecipeKeyMaskPage +
                                    key_range_scratch_bytes(key_range)
                              : 0;
}

static std::vector<Tensor> run_recipe_segments(
    const std::vector<std::array<Tensor, 3>>& segments,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config,
    const std::optional<Tensor>& attn_mask,
    std::optional<float> scale,
    const MemoryConfig& output_memory_config,
    const RecipeKeyRange& key_range = {},
    const RecipeDenseOptions& options = {}) {
    const bool keyed = key_range.active();
    const bool paged = key_range.page_table.has_value();
    const auto& [q, k, v] = segments.front();
    // MLA without a V tensor: V is K's first head_dim_v columns (the reader reads them from K).
    const bool v_is_k = v.buffer() == k.buffer();
    std::vector<Tensor> io;
    for (const auto& segment : segments) {
        io.insert(io.end(), segment.begin(), segment.end());
    }
    if (v_is_k) {
        TT_FATAL(segments.size() == 1, "SDPA recipes read V from K on the dense path only");
        io.pop_back();  // one binding per buffer (generic_op)
    }
    for (const auto& input : io) {
        const Tensor* tensor = &input;
        TT_FATAL(tensor->storage_type() == StorageType::DEVICE, "SDPA recipes require device inputs");
        TT_FATAL(tensor->device() == q.device(), "SDPA recipe inputs must belong to the same device");
        TT_FATAL(
            tensor->device()->arch() == tt::ARCH::BLACKHOLE || tensor->device()->arch() == tt::ARCH::WORMHOLE_B0,
            "SDPA recipes support Blackhole and Wormhole B0 only");
        TT_FATAL(tensor->layout() == Layout::TILE, "SDPA recipes require tiled inputs");
        TT_FATAL(
            tensor->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "SDPA recipes require interleaved (DRAM or L1) inputs");
        TT_FATAL(tensor->logical_shape().rank() == 4, "SDPA recipes require rank-four inputs");
        const auto& shape = tensor->logical_shape();
        const auto& padded = tensor->padded_shape();
        TT_FATAL(
            shape[0] == padded[0] && shape[1] == padded[1] && shape[3] == padded[3] &&
                padded[2] == ((shape[2] + 31) / 32) * 32,
            "SDPA recipes only support minimal sequence-axis tile padding");
        TT_FATAL(tensor->tensor_spec().tile() == Tile({32, 32}), "SDPA recipes require standard 32x32 tiles");
    }
    const auto& qs = q.logical_shape();
    const DataType kv_type = policy.selection.kv_storage == KVStorage::BF16   ? DataType::BFLOAT16
                             : policy.selection.kv_storage == KVStorage::BFP8 ? DataType::BFLOAT8_B
                                                                              : DataType::BFLOAT4_B;
    // MLA: V and the output are head_dim_v wide (V may be K itself); Q and K qs[3].
    const uint32_t head_dim_v = options.head_dim_v ? options.head_dim_v : qs[3];
    TT_FATAL(
        head_dim_v % 32 == 0 && head_dim_v > 0 && head_dim_v <= qs[3],
        "SDPA recipes require a tile-aligned V head dim no larger than Q's, got {} for Q {}",
        head_dim_v,
        qs[3]);
    TT_FATAL(segments.size() == 1 || head_dim_v == qs[3], "SDPA recipe joint segments require V's head dim to be Q's");
    // Paged K/V (chunked prefill): the cache's view of blocks per sequence x block size rows.
    const auto [block_rows, kv_heads] = paged
                                            ? paged_block_view(k, key_range)
                                            : std::pair<uint32_t, uint32_t>{k.logical_shape()[2], k.logical_shape()[1]};
    const uint32_t k_rows = recipe_k_rows(k, key_range);
    uint32_t q_length = 0, k_length = 0;
    for (const auto& [sq, sk, sv] : segments) {
        const auto& qshape = sq.logical_shape();
        const auto& kshape = sk.logical_shape();
        const auto& vshape = sv.logical_shape();
        TT_FATAL(
            qshape[0] > 0 && qshape[0] == qs[0] && qshape[1] > 0 && qshape[1] == qs[1] && qshape[3] == qs[3],
            "SDPA recipe segments require Q [B,H,Q,D] with matching positive batch/head counts");
        // Chunked prefill: K/V are [cache blocks, Hkv, block size, D] (RecipeKeyRange::page_table).
        TT_FATAL(
            (paged || kshape[0] == qs[0]) && kshape[1] > 0 && kshape[1] == k.logical_shape()[1] && kv_heads > 0 &&
                qs[1] % kv_heads == 0,
            "SDPA recipes require K/V [B,Hkv,K,D] with Q heads divisible by KV heads");
        if (key_range.paged_geometry.active()) {
            // A shared cache allocated for another layer: this call's view must cover each block exactly.
            TT_FATAL(
                head_dim_v == qs[3] && !v_is_k, "paged_cache_geometry is not supported with multi-latent attention");
            TT_FATAL(
                uint64_t{kshape[1]} * kshape[2] * kshape[3] == uint64_t{kv_heads} * block_rows * qs[3],
                "paged_cache_geometry (block size {}, {} KV heads) must hold as many elements per block as the cache "
                "{}",
                block_rows,
                kv_heads,
                kshape);
        } else {
            TT_FATAL(
                kshape[3] == qs[3], "SDPA recipes require K's head dim to be Q's, got {} and {}", kshape[3], qs[3]);
        }
        TT_FATAL(
            v_is_k || (vshape[0] == kshape[0] && vshape[1] == kshape[1] && vshape[2] == kshape[2] &&
                       vshape[3] == (key_range.paged_geometry.active() ? kshape[3] : head_dim_v)),
            "SDPA recipes require V {} to match K {} with head dim {}",
            vshape,
            kshape,
            head_dim_v);
        TT_FATAL(
            !paged || block_rows % 32 == 0, "SDPA recipe paged K/V blocks must be tile multiples, got {}", block_rows);
        TT_FATAL(qshape[2] > 0 && kshape[2] > 0, "SDPA recipe segments require positive sequence lengths");
        TT_FATAL(
            sq.dtype() == DataType::BFLOAT16 && sk.dtype() == kv_type && sv.dtype() == kv_type,
            "SDPA input types do not match the selected recipe");
        q_length += sq.padded_shape()[2];
        k_length += paged ? k_rows : sk.padded_shape()[2];
    }
    if (options.attention_sink) {
        const auto& sink = *options.attention_sink;
        TT_FATAL(segments.size() == 1, "SDPA recipe attention sinks are supported on the dense path only");
        TT_FATAL(
            sink.storage_type() == StorageType::DEVICE && sink.device() == q.device(),
            "SDPA recipe attention_sink must be on Q's device");
        TT_FATAL(
            sink.layout() == Layout::TILE && sink.tensor_spec().tile() == Tile({32, 32}) &&
                (sink.dtype() == DataType::BFLOAT16 || sink.dtype() == DataType::FLOAT32) && !sink.is_sharded(),
            "SDPA recipe attention_sink must be an interleaved BF16 or FP32 32x32-tiled tensor");
        const auto& shape = sink.logical_shape();
        TT_FATAL(
            shape.rank() == 4 && shape[0] == 1 && shape[1] == qs[1] && shape[2] == 1 && shape[3] == 1,
            "SDPA recipe attention_sink must be [1, {}, 1, 1], got {}",
            qs[1],
            shape);
    }
    TT_FATAL(
        segments.size() == 1 || !options.output_concat_heads,
        "SDPA recipe output_concat_heads is supported on the dense path only");
    if (attn_mask) {
        TT_FATAL(segments.size() == 1, "SDPA recipe masks are supported on the dense (non-joint) path only");
        validate_recipe_mask(q, k, *attn_mask, policy);
    }
    if (keyed) {
        TT_FATAL(segments.size() == 1, "SDPA recipe key ranges are supported on the dense (non-joint) path only");
        TT_FATAL(!attn_mask, "SDPA recipes take either an attn_mask or a causal / window / chunked key range");
        validate_key_range(q, key_range);
    }
    if (key_range.q_slab_rows) {
        // Ring-distributed SDPA computes two slabs of Q's rows (RecipeKeyRange::q_slab_rows).
        q_length = 2 * key_range.q_slab_rows;
    }
    const uint32_t joint_q_rows = segments.size() == 2 ? segments[1][0].logical_shape()[2] : 0;
    const uint32_t joint_k_rows = segments.size() == 2 ? segments[1][1].logical_shape()[2] : 0;
    const auto hardware = q.device()->compute_with_storage_grid_size();
    const auto grid_size = program_config ? program_config->compute_with_storage_grid_size : hardware;
    TT_FATAL(
        grid_size.x > 0 && grid_size.y > 0 && grid_size.x <= hardware.x && grid_size.y <= hardware.y,
        "SDPA recipe compute grid must fit the device");
    const uint32_t q_tiles = recipe_dense_q_tiles(program_config);
    const uint32_t q_chunk = q_tiles * 32;
    const uint32_t k_tiles = recipe_dense_k_tiles(program_config);
    const uint32_t k_chunk = k_tiles * 32;
    TT_FATAL(qs[3] % 32 == 0 && qs[3] > 0, "SDPA recipes support tile-aligned head dims, got {}", qs[3]);
    const uint32_t d_tiles = qs[3] / 32;
    const uint32_t vd_tiles = head_dim_v / 32;
    TT_FATAL(
        key_range.q_slab_rows % q_chunk == 0,
        "SDPA recipe Q slabs of {} rows must hold whole Q chunks of {} rows",
        key_range.q_slab_rows,
        q_chunk);
    // Reader runtime args 16-17 hold the slabs (kRecipeReaderSlabArg), where a keyed call's sink address would go.
    TT_FATAL(!key_range.q_slab_rows || !options.attention_sink, "SDPA recipe Q slabs do not take an attention sink");
    if (program_config) {
        TT_FATAL(!program_config->sub_core_grids.has_value(), "SDPA recipes do not yet support sub_core_grids");
        TT_FATAL(program_config->max_cores_per_head_batch > 0, "SDPA max_cores_per_head_batch must be positive");
    }
    const uint32_t jobs_per_head = (q_length + q_chunk - 1) / q_chunk;
    const uint32_t k_chunks = (k_length + k_chunk - 1) / k_chunk;
    const uint32_t batch_heads = qs[0] * qs[1];
    const uint32_t grid_cores = grid_size.x * grid_size.y;
    // Up to one batch/head per core: a K/V-forwarding chain of up to max_cores_per_head_batch cores per head.
    // More batch/heads than cores: every Q chunk of every head is one job, split evenly over the grid without
    // chains (a core's jobs may span heads; the reader follows each job's head).
    const bool global_jobs = batch_heads > grid_cores;
    const uint32_t chain = global_jobs ? 1u
                                       : std::min<uint32_t>(
                                             {jobs_per_head,
                                              grid_cores / batch_heads,
                                              program_config ? program_config->max_cores_per_head_batch : 16u});
    TT_FATAL(chain > 0, "SDPA recipes require at least one compute core per head");
    const uint32_t total_jobs = batch_heads * jobs_per_head;
    const uint32_t cores = global_jobs ? std::min(grid_cores, total_jobs) : chain * batch_heads;
    std::vector<CoreCoord> coordinates;
    std::set<CoreRange> ranges;
    for (uint32_t i = 0; i < cores; ++i) {
        const CoreCoord core(i % grid_size.x, i / grid_size.x);
        coordinates.push_back(core);
        ranges.emplace(core, core);
    }
    const CoreRangeSet grid(ranges);
    TT_FATAL(
        output_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "SDPA recipes require an interleaved (DRAM or L1) output memory config");
    std::vector<Tensor> outputs;
    for (const auto& segment : segments) {
        auto shape = segment[0].logical_shape();
        shape[3] = head_dim_v;
        if (key_range.q_slab_rows) {
            shape[2] = 2 * key_range.q_slab_rows;
        }
        if (options.output_concat_heads) {
            shape = ttnn::Shape({shape[0], 1, shape[2], shape[1] * shape[3]});
        }
        outputs.push_back(create_device_tensor(
            TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), output_memory_config)),
            q.device()));
    }
    const auto& output = outputs.front();
    const uint32_t compute_q_tiles =
        recipe_compute_q_tiles(policy, q_tiles, k_tiles, attn_mask.has_value() || keyed, keyed);
    auto program = recipe_compute_program(policy, grid, k_chunks, compute_q_tiles, k_tiles, d_tiles, scale, vd_tiles);
    // QK row-group height the compute consumes the mask in: FP32 recipes single rows, paired BF16 recipes row pairs.
    const uint32_t mask_group_rows = policy.fp32_destination ? 1 : 2;
    if (attn_mask || keyed) {
        // The reader streams mask tiles one Q tile row (k_tiles tiles) at a time in whole row groups
        // (an odd paired chunk's last group is padded with a zero row), and compute pops one group at a
        // time, so group reads never wrap. Double-buffer the group when L1 allows, else single.
        // Key ranges generate BFP4 {0, -2^100} tiles for their edge chunks only.
        const auto mask_format =
            attn_mask ? datatype_to_dataformat_converter(attn_mask->dtype()) : tt::DataFormat::Bfp4_b;
        const uint32_t mask_page = attn_mask ? attn_mask->buffer()->page_size() : kRecipeKeyMaskPage;
        const uint32_t group_bytes = mask_group_rows * k_tiles * mask_page;
        uint64_t used = recipe_key_range_extra_bytes(key_range) + recipe_dense_options_extra_bytes(options);
        for (const auto& cb : program.cbs) {
            used += cb.total_size;
        }
        const uint64_t available = recipe_free_l1(*q.device());
        const uint32_t groups = used + 2 * group_bytes <= available ? 2 : 1;
        program.cbs.push_back(CBDescriptor{
            .total_size = groups * group_bytes,
            .core_ranges = grid,
            .format_descriptors = {{.buffer_index = kRecipeMaskCb, .data_format = mask_format, .page_size = mask_page}}});
        auto& defines = program.kernels.front().defines;
        defines.emplace_back("SDPA_RECIPE_MASK", "1");
        if (keyed) {
            defines.emplace_back("SDPA_RECIPE_KRANGE", "1");
        }
        if (mask_format == tt::DataFormat::Float32) {
            // Unpack the FP32 mask straight to DST so the L1 add sees the exact FP32 values.
            auto& config = std::get<ComputeConfigDescriptor>(program.kernels.front().config);
            config.unpack_to_dest_mode[kRecipeMaskCb] = UnpackToDestMode::UnpackToDestFp32;
        }
    }
    if (keyed) {
        auto add_cb = [&](uint8_t index, uint32_t pages, uint32_t page, tt::DataFormat format) {
            program.cbs.push_back(CBDescriptor{
                .total_size = pages * page,
                .core_ranges = grid,
                .format_descriptors = {{.buffer_index = index, .data_format = format, .page_size = page}}});
        };
        add_cb(kRecipeKeyRangeCb, 2, kRecipeKeyRangePage, tt::DataFormat::UInt32);
        add_cb(kRecipeMaskedTileCb, 1 + kRecipeMaskCacheTiles, kRecipeKeyMaskPage, tt::DataFormat::Bfp4_b);
        if (const uint32_t scratch = key_range_scratch_bytes(key_range)) {
            add_cb(kRecipeKeyScratchCb, 1, scratch, tt::DataFormat::UInt32);
        }
    }
    if (options.attention_sink) {
        // One page per Q chunk: the head's sink logit (reader), read by compute at normalization.
        program.cbs.push_back(CBDescriptor{
            .total_size = 2 * kRecipeSinkPage,
            .core_ranges = grid,
            .format_descriptors = {
                {.buffer_index = kRecipeSinkCb, .data_format = tt::DataFormat::UInt32, .page_size = kRecipeSinkPage}}});
        auto& defines = program.kernels.front().defines;
        defines.emplace_back("SDPA_RECIPE_SINK", std::to_string(kRecipeSinkCb));
        if (options.attention_sink->dtype() == DataType::BFLOAT16) {
            defines.emplace_back("SDPA_RECIPE_SINK_BF16", "1");
        }
    }
    check_recipe_l1_fit(program, *q.device(), q_chunk, k_chunk);
    if (k_length % k_chunk != 0 || k_rows % 32 != 0 || joint_k_rows % 32 != 0) {
        program.kernels.front().defines.emplace_back("SDPA_RECIPE_K_PRIMARY_ROWS", std::to_string(k_rows));
        program.kernels.front().defines.emplace_back("SDPA_RECIPE_K_JOINT_ROWS", std::to_string(joint_k_rows));
    }
    for (uint32_t i = 0; i < 3; ++i) {
        program.semaphores.push_back({.id = i, .core_ranges = grid, .initial_value = i == 2 ? 1u : 0u});
    }
    const std::string prefix = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    KernelDescriptor reader{
        .kernel_source = prefix + "dataflow/reader_recipe.cpp",
        .core_ranges = grid,
        .compile_time_args =
            {q_tiles, k_chunks, jobs_per_head, q.logical_shape()[2], joint_q_rows, k_rows, joint_k_rows},
        .config = ReaderConfigDescriptor{}};
    if (segments.size() == 2) {
        reader.defines.emplace_back("SDPA_JOINT", "1");
    }
    reader.defines.emplace_back("SDPA_K_CHUNK_TILES", std::to_string(k_tiles));
    reader.defines.emplace_back("SDPA_RECIPE_Q_PAD_TILES", std::to_string(compute_q_tiles - q_tiles));
    reader.defines.emplace_back("SDPA_RECIPE_DHT", std::to_string(d_tiles));
    if (qs[1] != kv_heads) {
        reader.defines.emplace_back("SDPA_RECIPE_Q_PER_KV_HEAD", std::to_string(qs[1] / kv_heads));
    }
    if (paged || vd_tiles != d_tiles || v_is_k) {
        // K/V tile rows located one at a time (dataflow/reader_recipe.cpp: read_kv_rows): through the page table, and
        // for MLA a V narrower than its source rows (V's own tensor, or K's first head_dim_v columns).
        reader.defines.emplace_back("SDPA_RECIPE_KV_ROWS", std::to_string(k_rows));
        reader.defines.emplace_back("SDPA_RECIPE_V_DHT", std::to_string(vd_tiles));
        reader.defines.emplace_back("SDPA_RECIPE_V_SRC_DHT", std::to_string(v_is_k ? d_tiles : vd_tiles));
        if (v_is_k) {
            reader.defines.emplace_back("SDPA_RECIPE_V_IS_K", "1");
        }
    }
    for (const auto& segment : segments) {
        for (uint32_t i = 0; i < (v_is_k ? 2u : 3u); ++i) {
            TensorAccessorArgs(segment[i].buffer()).append_to(reader.compile_time_args);
        }
    }
    if (attn_mask) {
        const auto& ms = attn_mask->logical_shape();
        reader.defines.emplace_back("SDPA_RECIPE_MASK", "1");
        reader.defines.emplace_back("SDPA_RECIPE_MASK_CB", std::to_string(kRecipeMaskCb));
        reader.defines.emplace_back("SDPA_RECIPE_MASK_GROUP_ROWS", std::to_string(mask_group_rows));
        reader.defines.emplace_back("SDPA_RECIPE_MASK_Q_TILES", std::to_string((ms[2] + 31) / 32));
        reader.defines.emplace_back("SDPA_RECIPE_MASK_K_TILES", std::to_string((ms[3] + 31) / 32));
        reader.defines.emplace_back("SDPA_RECIPE_MASK_HEADS", std::to_string(qs[1]));
        reader.defines.emplace_back("SDPA_RECIPE_MASK_BCAST_BATCH", ms[0] == 1 ? "1" : "0");
        reader.defines.emplace_back("SDPA_RECIPE_MASK_BCAST_HEADS", ms[1] == 1 ? "1" : "0");
        TensorAccessorArgs(attn_mask->buffer()).append_to(reader.compile_time_args);
    }
    KernelDescriptor writer{
        .kernel_source = prefix + "dataflow/writer_recipe.cpp",
        .core_ranges = grid,
        .compile_time_args = {q_tiles, q.logical_shape()[2], joint_q_rows},
        .config = WriterConfigDescriptor{}};
    if (segments.size() == 2) {
        writer.defines.emplace_back("SDPA_JOINT", "1");
    }
    writer.defines.emplace_back("SDPA_RECIPE_DHT", std::to_string(vd_tiles));
    if (options.output_concat_heads) {
        writer.defines.emplace_back("SDPA_RECIPE_CONCAT_HEADS", std::to_string(qs[1]));
        if (!keyed) {
            writer.defines.emplace_back("SDPA_RECIPE_Q_JOBS", std::to_string(jobs_per_head));
        }
    }
    writer.defines.emplace_back("SDPA_RECIPE_Q_PAD_TILES", std::to_string(compute_q_tiles - q_tiles));
    for (const auto& tensor : outputs) {
        TensorAccessorArgs(tensor.buffer()).append_to(writer.compile_time_args);
    }
    // Key ranges: the reader streams each Q chunk's K range; the writer sends compute the range and generates the
    // edge masks (dataflow/recipe_key_range.hpp). Both read the Q offset and cu_window_seqlens tensors; the reader
    // also reads the page table. Device tensors are bound in the order Q offset, cu_window_seqlens, page table.
    std::vector<Tensor> key_tensors;
    if (keyed) {
        TT_FATAL(
            !key_range.q_offset_tensor || key_range.q_offset_tensor->buffer()->aligned_page_size() <= 64,
            "SDPA recipe Q offset tensor page must fit 64 bytes");
        const std::string segment_bounds =
            std::to_string(key_range.segments ? key_range.segments->logical_shape()[0] : 0);
        for (auto* kernel : {&reader, &writer}) {
            kernel->defines.emplace_back("SDPA_RECIPE_KRANGE", "1");
            kernel->defines.emplace_back("SDPA_RECIPE_CAUSAL", key_range.causal ? "1" : "0");
            kernel->defines.emplace_back("SDPA_RECIPE_WINDOW", std::to_string(key_range.sliding_window));
            kernel->defines.emplace_back("SDPA_RECIPE_SEGMENTS", segment_bounds);
            kernel->defines.emplace_back("SDPA_RECIPE_SCRATCH_CB", std::to_string(kRecipeKeyScratchCb));
        }
        writer.defines.emplace_back("SDPA_RECIPE_Q_JOBS", std::to_string(jobs_per_head));
        if (key_range.q_slab_rows) {
            for (auto* kernel : {&reader, &writer}) {
                kernel->defines.emplace_back("SDPA_RECIPE_Q_SLAB_JOBS", std::to_string(key_range.q_slab_rows / q_chunk));
            }
        }
        writer.defines.emplace_back("SDPA_RECIPE_K_ROWS", std::to_string(k_rows));
        writer.defines.emplace_back("SDPA_RECIPE_K_CHUNKS", std::to_string(k_chunks));
        writer.defines.emplace_back("SDPA_K_CHUNK_TILES", std::to_string(k_tiles));
        writer.defines.emplace_back("SDPA_RECIPE_MASK_CB", std::to_string(kRecipeMaskCb));
        writer.defines.emplace_back("SDPA_RECIPE_MASK_GROUP_ROWS", std::to_string(mask_group_rows));
        writer.defines.emplace_back("SDPA_RECIPE_KEY_RANGE_CB", std::to_string(kRecipeKeyRangeCb));
        writer.defines.emplace_back("SDPA_RECIPE_MASKED_TILE_CB", std::to_string(kRecipeMaskedTileCb));
        writer.defines.emplace_back("SDPA_RECIPE_MASK_CACHE_TILES", std::to_string(kRecipeMaskCacheTiles));
        writer.defines.emplace_back("SDPA_RECIPE_SCRATCH_WRITER", std::to_string(key_range_scratch_half(key_range)));
        for (const auto& [tensor, define, to_writer] :
             {std::tuple{&key_range.q_offset_tensor, "SDPA_RECIPE_Q_OFFSET_PAGE", true},
              std::tuple{&key_range.segments, "SDPA_RECIPE_SEGMENTS_PAGE", true},
              std::tuple{&key_range.page_table, "SDPA_RECIPE_PAGE_TABLE_PAGE", false}}) {
            if (!*tensor) {
                continue;
            }
            key_tensors.push_back(**tensor);
            const auto page = std::to_string((*tensor)->buffer()->aligned_page_size());
            for (auto* kernel : to_writer ? std::vector{&reader, &writer} : std::vector{&reader}) {
                kernel->defines.emplace_back(define, page);
            }
        }
        // Without the chain every core reads its own K/V: bound the reads in flight per core like legacy SDPA
        // (dataflow_common.hpp: get_barrier_read_threshold), at most the chain head's 16 tiles. Measured on 10 heads x
        // 8192^2 causal over 110 cores: 16 tiles 2.66 ms, 2 tiles 2.25 ms (STANDARD).
        const uint32_t kv_page = k.buffer()->page_size();
        reader.defines.emplace_back(
            "SDPA_RECIPE_READ_BARRIER_TILES",
            std::to_string(std::clamp<uint32_t>((512 / cores) * (1024 + 128) / kv_page, 1, 16)));
        if (paged) {
            reader.defines.emplace_back("SDPA_RECIPE_Q_HEADS", std::to_string(qs[1]));
            reader.defines.emplace_back("SDPA_RECIPE_KV_HEADS", std::to_string(kv_heads));
            reader.defines.emplace_back("SDPA_RECIPE_PAGE_BLOCK_TILES", std::to_string(block_rows / 32));
            reader.defines.emplace_back(
                "SDPA_RECIPE_PAGE_TABLE_OFFSET", std::to_string(2 * key_range_scratch_half(key_range)));
        }
        for (const auto& tensor : key_tensors) {
            TensorAccessorArgs(tensor.buffer()).append_to(reader.compile_time_args);
        }
        for (const auto& tensor : {key_range.q_offset_tensor, key_range.segments}) {
            if (tensor) {
                TensorAccessorArgs(tensor->buffer()).append_to(writer.compile_time_args);
            }
        }
    }
    if (options.attention_sink) {
        // The sink tensor's accessor and address come last (runtime arg SDPA_RECIPE_SINK_ARG).
        reader.defines.emplace_back("SDPA_RECIPE_SINK_CB", std::to_string(kRecipeSinkCb));
        reader.defines.emplace_back("SDPA_RECIPE_SINK_HEADS", std::to_string(qs[1]));
        reader.defines.emplace_back("SDPA_RECIPE_SINK_CTA", std::to_string(reader.compile_time_args.size()));
        reader.defines.emplace_back("SDPA_RECIPE_SINK_ARG", std::to_string(keyed ? 16 : (attn_mask ? 13 : 12)));
        TensorAccessorArgs(options.attention_sink->buffer()).append_to(reader.compile_time_args);
    }
    const uint32_t sink_address = options.attention_sink ? options.attention_sink->buffer()->address() : 0;
    auto compute = std::move(program.kernels.front());
    // Key ranges: no K/V chain (each Q chunk reads its own K range), Q chunks in zigzag order.
    const auto zigzag =
        keyed ? recipe_zigzag_split(jobs_per_head, chain) : std::vector<std::pair<uint32_t, uint32_t>>{};
    for (uint32_t i = 0; i < cores; ++i) {
        const auto core = coordinates[i];
        const uint32_t head = i / chain, rank = i % chain;
        // Jobs of one head split over its chain, or (global_jobs, chain 1) all jobs split over the cores.
        const uint32_t split_jobs = global_jobs ? total_jobs : jobs_per_head;
        const uint32_t split_ways = global_jobs ? cores : chain;
        const uint32_t part = global_jobs ? i : rank;
        const uint32_t count = split_jobs / split_ways + (part < split_jobs % split_ways);
        const uint32_t offset = (global_jobs ? 0 : head * jobs_per_head) + part * (split_jobs / split_ways) +
                                std::min(part, split_jobs % split_ways);
        if (keyed) {
            // A contiguous range of positions in the heads' zigzag orders (position z: head z / J, its zigzag
            // entry z % J): the chain's zigzag split of one head, or with more batch/heads than cores the even split
            // of all of them.
            const uint32_t z_first = global_jobs ? offset : head * jobs_per_head + zigzag[rank].first;
            const uint32_t z_count = global_jobs ? count : zigzag[rank].second;
            reader.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    q.buffer()->address(),
                    k.buffer()->address(),
                    v.buffer()->address(),
                    z_first,
                    z_count,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    key_range.q_offset});
            for (const auto& tensor : {key_range.q_offset_tensor, key_range.segments, key_range.page_table}) {
                reader.runtime_args.back().second.push_back(tensor ? tensor->buffer()->address() : 0);
            }
            if (options.attention_sink) {
                reader.runtime_args.back().second.push_back(sink_address);
            }
            writer.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    output.buffer()->address(),
                    z_first,
                    z_count,
                    key_range.q_offset,
                    key_range.q_offset_tensor ? key_range.q_offset_tensor->buffer()->address() : 0,
                    key_range.segments ? key_range.segments->buffer()->address() : 0});
            if (key_range.q_slab_rows) {
                // Reader args 16-17, writer args 6-7: the slabs' first Q chunks, set per device below.
                reader.runtime_args.back().second.resize(kRecipeReaderSlabArg + 2);
                writer.runtime_args.back().second.resize(kRecipeWriterSlabArg + 2);
            }
            compute.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{z_count});
            continue;
        }
        const auto prev = rank ? q.device()->worker_core_from_logical_core(coordinates[i - 1]) : CoreCoord(0, 0);
        const auto next =
            rank + 1 < chain ? q.device()->worker_core_from_logical_core(coordinates[i + 1]) : CoreCoord(0, 0);
        const uint32_t next_count = rank + 1 < chain ? jobs_per_head / chain + (rank + 1 < jobs_per_head % chain) : 0;
        reader.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                q.buffer()->address(),
                k.buffer()->address(),
                v.buffer()->address(),
                offset,
                count,
                rank,
                chain,
                prev.x,
                prev.y,
                next.x,
                next.y,
                next_count});
        writer.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{output.buffer()->address(), offset, count});
        if (attn_mask) {
            reader.runtime_args.back().second.push_back(attn_mask->buffer()->address());
        }
        if (options.attention_sink) {
            reader.runtime_args.back().second.push_back(sink_address);
        }
        if (segments.size() == 2) {
            for (const auto& tensor : segments[1]) {
                reader.runtime_args.back().second.push_back(tensor.buffer()->address());
            }
            writer.runtime_args.back().second.push_back(outputs[1].buffer()->address());
        }
        compute.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{count});
    }
    program.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    if (attn_mask) {
        io.push_back(*attn_mask);
    }
    if (options.attention_sink) {
        io.push_back(*options.attention_sink);
    }
    io.insert(io.end(), key_tensors.begin(), key_tensors.end());
    io.insert(io.end(), outputs.begin(), outputs.end());
    if (keyed) {
        // The scalar Q offset is a runtime arg the cache-hit path does not re-apply: key the program on it (legacy
        // chunked SDPA keys its cache on chunk_start_idx the same way); the kernels do not recompile.
        auto hash = ttnn::operations::generic::compute_program_descriptor_hash(program);
        ttsl::hash::hash_combine(hash, key_range.q_offset);
        program.custom_program_hash = hash;
    }
    if (!key_range.q_slab_rows) {
        ttnn::generic_op(io, program);
        return outputs;
    }
    // Ring-distributed SDPA: one program per range of devices, differing only in the Q slabs' first chunks (runtime
    // args, so folded into the program hash like the Q offset).
    tt::tt_metal::experimental::MeshProgramDescriptor mesh_program;
    for (const auto& [devices, starts] : key_range.q_slab_starts) {
        auto device_program = program;
        for (auto& [core, args] : device_program.kernels[0].runtime_args) {
            args[kRecipeReaderSlabArg] = starts[0] / q_chunk;
            args[kRecipeReaderSlabArg + 1] = starts[1] / q_chunk;
        }
        for (auto& [core, args] : device_program.kernels[1].runtime_args) {
            args[kRecipeWriterSlabArg] = starts[0] / q_chunk;
            args[kRecipeWriterSlabArg + 1] = starts[1] / q_chunk;
        }
        auto hash = *program.custom_program_hash;
        ttsl::hash::hash_combine(hash, starts[0]);
        ttsl::hash::hash_combine(hash, starts[1]);
        device_program.custom_program_hash = hash;
        mesh_program.mesh_programs.emplace_back(devices, std::move(device_program));
    }
    ttnn::generic_op(io, mesh_program);
    return outputs;
}

Tensor run_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config,
    const std::optional<Tensor>& attn_mask,
    std::optional<float> scale,
    const MemoryConfig& output_memory_config,
    const RecipeKeyRange& key_range,
    const RecipeDenseOptions& options) {
    return run_recipe_segments(
               {{q, k, v}}, policy, program_config, attn_mask, scale, output_memory_config, key_range, options)
        .front();
}

std::tuple<Tensor, Tensor> run_joint_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& joint_q,
    const Tensor& joint_k,
    const Tensor& joint_v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config,
    std::optional<float> scale) {
    auto outputs = run_recipe_segments(
        {{q, k, v}, {joint_q, joint_k, joint_v}}, policy, program_config, std::nullopt, scale, DRAM_MEMORY_CONFIG);
    return {outputs[0], outputs[1]};
}

Tensor recipe_bf16_query(const Tensor& q) {
    if (q.dtype() == DataType::BFLOAT16) {
        return q;
    }
    TT_FATAL(
        q.dtype() == DataType::BFLOAT8_B || q.dtype() == DataType::BFLOAT4_B,
        "SDPA recipes take a BF16, BFP8 or BFP4 Q, got {}",
        q.dtype());
    return ttnn::typecast(q, DataType::BFLOAT16, DRAM_MEMORY_CONFIG);
}

uint64_t recipe_output_l1_bytes(
    const Tensor& q, uint64_t pages, uint32_t page_bytes, const MemoryConfig& memory_config) {
    if (memory_config.buffer_type() != BufferType::L1) {
        return 0;
    }
    const uint64_t banks = q.device()->allocator()->get_num_banks(BufferType::L1);
    const uint64_t alignment = q.device()->allocator()->get_alignment(BufferType::L1);
    const uint64_t bytes = ((pages + banks - 1) / banks) * page_bytes;
    return (bytes + alignment - 1) / alignment * alignment;
}

}  // namespace ttnn::operations::transformer::sdpa::detail
