// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <string_view>
#include <utility>

#include <tt-logger/tt-logger.hpp>

#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa_recipe.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa_recipe_blocking.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_distributed_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"

namespace ttnn::transformer {

namespace {
// Empty joint (0 elements) == "no joint". Normalize to nullopt so self-attention callers passing
// zero-length dummy joints don't create duplicate input Buffer*s -> resolve_bindings() bails and the
// WorkloadDescriptor cache-hit path freezes stale addresses (#45452 / #45391). L=0 either way, so
// numerically identical.
std::optional<ttnn::Tensor> drop_if_empty(const std::optional<ttnn::Tensor>& t) {
    if (t.has_value() && t->logical_shape().volume() == 0) {
        return std::nullopt;
    }
    return t;
}

// Precision routing (tech_reports/FlashAttention/SDPAPrecisionRecipes.md, "Routing"). A call without `precision`
// that would reach a legacy compute_common.hpp loop runs a recipe instead: ACCURATE when its compute config asks
// for FP32 DEST accumulation, STANDARD on the routes that have no streaming kernel (non-ring joint, exp ring
// blockings its streaming kernel cannot build). BF16-dest dense, chunked and ring calls keep the streaming kernels
// (compute_streaming.hpp). Routed calls treat program_config chunk sizes as hints (they were chosen for the legacy
// kernels): kept when the recipe supports them and they fit, otherwise the op chooses the blocking.
// TODO(SDPA recipes on Wormhole): the recipes run on Blackhole only, so Wormhole keeps the legacy loops until the
// Wormhole port lands; remove this arch gate (one use per entry point) with it.
bool routes_to_recipes(const ttnn::Tensor& q) {
    return q.storage_type() == StorageType::DEVICE && q.device()->arch() == tt::ARCH::BLACKHOLE;
}

// The compute config the legacy kernels would run with (the defaults every SDPA prefill op applies).
DeviceComputeKernelConfig legacy_compute_config(
    const ttnn::Tensor& q, const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return init_device_compute_kernel_config(
        q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);
}

// The recipe a call without `precision` runs on routes that always leave the streaming kernels.
SDPAPrecision routed_precision(
    const ttnn::Tensor& q, const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return get_fp32_dest_acc_en(legacy_compute_config(q, compute_kernel_config)) ? SDPAPrecision::ACCURATE
                                                                                 : SDPAPrecision::STANDARD;
}

// Dense, chunked, MLA prefill and ring joint: only FP32 DEST reaches a legacy loop (the BF16 path streams), and
// runs ACCURATE.
bool routes_fp32_dest(const ttnn::Tensor& q, const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return routes_to_recipes(q) && get_fp32_dest_acc_en(legacy_compute_config(q, compute_kernel_config));
}

// A routed call's program config: legacy prefill ignores sub_core_grids (only decode reads it), so drop it.
std::optional<ttnn::operations::transformer::SDPAProgramConfig> routed_program_config(
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config) {
    if (program_config) {
        program_config->sub_core_grids = std::nullopt;
    }
    return program_config;
}

// Ring and exp ring: the caller's chunk sizes unless the recipe does not support that geometry, in which case
// both are left to the op's blocking chooser.
void drop_unsupported_routed_chunks(
    ttnn::operations::transformer::SDPAProgramConfig& program_config,
    operations::transformer::sdpa::detail::RecipeOp op,
    const operations::transformer::sdpa::detail::PrecisionPolicy& policy,
    uint32_t head_dim) {
    namespace numeric = operations::transformer::sdpa::detail;
    const bool tile_aligned = program_config.q_chunk_size % 32 == 0 && program_config.k_chunk_size % 32 == 0 &&
                              program_config.q_chunk_size > 0 && program_config.k_chunk_size > 0;
    if (!tile_aligned || head_dim % 32 != 0 ||
        !numeric::recipe_geometry_supported(
            op, policy, program_config.q_chunk_size / 32, program_config.k_chunk_size / 32, head_dim / 32)) {
        program_config.q_chunk_size = 0;
        program_config.k_chunk_size = 0;
    }
}

// Why ring_joint_scaled_dot_product_attention cannot run the ring recipe yet (nullopt: it can). A routed FP32-DEST
// call with one of these features keeps the legacy sdpa_ring loop until the ring recipe gains it; none has an FP32
// caller in models/. The legacy loop cannot be deleted while this returns anything.
std::optional<std::string_view> ring_recipe_gap(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    bool is_cross,
    const std::optional<ttnn::Tensor>& attention_sink,
    std::optional<uint32_t> sliding_window_size,
    bool circular_kv_cache,
    std::optional<uint32_t> kv_cache_batch_idx,
    std::optional<uint32_t> kv_actual_isl,
    const std::optional<ttnn::Tensor>& slot_id,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor) {
    if (attention_sink || sliding_window_size.value_or(0) > 0) {
        return "attention sink / sliding window";
    }
    if (circular_kv_cache || kv_cache_batch_idx || kv_actual_isl || slot_id || kv_actual_isl_tensor) {
        return "indexed / padded KV cache";
    }
    if (k.logical_shape()[3] != q.logical_shape()[3] || v.logical_shape()[3] != q.logical_shape()[3]) {
        return "V head dim != Q head dim";
    }
    if (!is_cross && q.logical_shape()[2] != k.logical_shape()[2]) {
        return "chunked prefill";
    }
    return std::nullopt;
}

// The recipe path of scaled_dot_product_attention, chunked_scaled_dot_product_attention and the MLA prefills: an
// optional additive attn_mask or a key range (causal, sliding window, chunked prefill, windowed), never both, plus the
// dense options (MLA V head dim, attention sink, concatenated-heads output).
ttnn::Tensor dense_recipe(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    SDPAPrecision precision,
    const operations::transformer::sdpa::detail::RecipeKeyRange& key_range,
    operations::transformer::sdpa::detail::RecipeDenseOptions options = {},
    bool routed = false) {
    namespace numeric = operations::transformer::sdpa::detail;
    const auto recipe_program_config = routed ? routed_program_config(program_config) : program_config;
    const auto output_memory_config = memory_config.value_or(DRAM_MEMORY_CONFIG);
    TT_FATAL(
        output_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "SDPA recipes require an interleaved (DRAM or L1) output memory config");
    const auto policy = numeric::resolve_recipe_policy(
        input_tensor_q, input_tensor_k, precision, scale, compute_kernel_config, recipe_program_config);
    const float recipe_scale = scale.value_or(1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1])));
    // Same mask contract as legacy SDPA: the recipe kernels fold the softmax scale into
    // the exponent, so the additive mask is pre-multiplied by 1/scale (0 and -inf are exact).
    // FP32-state recipes (BALANCED/ACCURATE) hold FP32 scores, so their mask is pre-scaled in FP32
    // (and added exactly); BF16-score recipes keep the legacy mask-dtype pre-scale. A pre-scale by a power of two
    // (scale 1 included) is exact in the mask's own format, so that mask stays narrow (half the mask traffic).
    std::optional<ttnn::Tensor> recipe_mask = attn_mask;
    if (attn_mask) {
        // Reject an unsupported mask before the pre-scale dispatches anything.
        numeric::validate_recipe_mask(input_tensor_q, input_tensor_k, *attn_mask, policy);
        int exponent = 0;
        const bool exact_prescale = std::frexp(1.0f / recipe_scale, &exponent) == 0.5f;
        if (policy.fp32_destination && attn_mask->dtype() != DataType::FLOAT32 && !exact_prescale) {
            recipe_mask = ttnn::typecast(*attn_mask, DataType::FLOAT32);
        }
        if (recipe_scale != 1.0f) {
            recipe_mask = ttnn::multiply(*recipe_mask, 1.0f / recipe_scale);
        }
    }
    // The kernels read BF16 Q and write BF16; a BFP8/BFP4 Q is widened first (exactly) and the output comes
    // back in Q's dtype like legacy SDPA. The BF16 intermediate then stays in DRAM.
    const auto query = numeric::recipe_bf16_query(input_tensor_q);
    const bool narrow_output = input_tensor_q.dtype() != DataType::BFLOAT16;
    // The reader takes the sink logit straight from a BF16 or FP32 tile; a BFP8/BFP4 sink is widened first.
    if (options.attention_sink && options.attention_sink->dtype() != DataType::BFLOAT16 &&
        options.attention_sink->dtype() != DataType::FLOAT32) {
        options.attention_sink = ttnn::typecast(*options.attention_sink, DataType::BFLOAT16, DRAM_MEMORY_CONFIG);
    }
    const auto kernel_memory_config = narrow_output ? DRAM_MEMORY_CONFIG : output_memory_config;
    const auto& qs = input_tensor_q.padded_shape();
    const uint32_t out_head_dim = options.head_dim_v ? options.head_dim_v : qs[3];
    // Op-selected blocking when program_config leaves chunks unset; the chooser budgets the
    // mask CB (at least one row group) so a masked call never picks a blocking that overflows L1,
    // and an L1 output's share of each core's L1 (it is allocated after the choice).
    const auto blocking = numeric::resolve_dense_recipe_blocking(
        policy,
        query,
        input_tensor_k,
        nullptr,
        nullptr,
        recipe_program_config,
        recipe_mask ? &*recipe_mask : nullptr,
        numeric::recipe_output_l1_bytes(
            query, uint64_t{qs[0]} * qs[1] * (qs[2] / 32) * (out_head_dim / 32), 2048, kernel_memory_config),
        &key_range,
        &options,
        routed);
    auto output = numeric::run_recipe(
        query,
        input_tensor_k,
        input_tensor_v,
        policy,
        blocking,
        recipe_mask,
        recipe_scale,
        kernel_memory_config,
        key_range,
        options);
    return narrow_output ? ttnn::typecast(output, input_tensor_q.dtype(), output_memory_config) : output;
}

// Chunked prefill on a named recipe: causal with the Q chunk at chunk_start_idx (scalar, or read on device from
// chunk_start_idx_tensor), over the paged K/V cache blocks the page table names. MLA (head_dim_v) reads V from K.
ttnn::Tensor chunked_recipe(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& page_table,
    std::optional<int64_t> chunk_start_idx,
    const std::optional<ttnn::Tensor>& chunk_start_idx_tensor,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::operations::transformer::PagedCacheGeometryOverride>& paged_cache_geometry,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<ttnn::Tensor>& attention_sink,
    SDPAPrecision precision,
    uint32_t head_dim_v = 0,
    bool routed = false) {
    namespace numeric = operations::transformer::sdpa::detail;
    TT_FATAL(q.storage_type() == StorageType::DEVICE, "SDPA recipes require device inputs");
    const numeric::RecipeKeyRange key_range{
        .causal = true,
        .sliding_window = sliding_window_size.value_or(0),
        .q_offset = static_cast<uint32_t>(chunk_start_idx.value_or(0)),
        .q_offset_tensor = chunk_start_idx_tensor,
        .page_table = page_table,
        .paged_geometry = paged_cache_geometry.value_or(operations::transformer::PagedCacheGeometryOverride{})};
    if (chunk_start_idx) {
        const uint32_t k_rows = numeric::recipe_k_rows(k, key_range);
        TT_FATAL(*chunk_start_idx >= 0, "chunk_start_idx must be non-negative");
        TT_FATAL(
            k_rows >= q.logical_shape()[2] + *chunk_start_idx,
            "K's sequence length must be >= Q's sequence length + chunk_start_idx. Got K: {}, Q: {}, chunk_start_idx: "
            "{}",
            k_rows,
            q.logical_shape()[2],
            *chunk_start_idx);
    }
    return dense_recipe(
        q,
        k,
        v,
        std::nullopt,
        scale,
        memory_config,
        program_config,
        compute_kernel_config,
        precision,
        key_range,
        {.head_dim_v = head_dim_v, .attention_sink = attention_sink},
        routed);
}

// Ring-distributed SDPA on a named recipe: each device computes the causal attention of two slabs of the whole Q,
// sequence chunks ring_id and 2 * ring_size - 1 - ring_id (RecipeKeyRange::q_slab_rows), with the legacy op's argument
// rules. ring_id, when not given, is the device's index along the mesh axis of length ring_size.
ttnn::Tensor ring_distributed_recipe(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    uint32_t ring_size,
    std::optional<uint32_t> ring_id,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& page_table,
    std::optional<int64_t> chunk_start_idx,
    SDPAPrecision precision,
    bool routed = false) {
    namespace numeric = operations::transformer::sdpa::detail;
    TT_FATAL(q.storage_type() == StorageType::DEVICE, "SDPA recipes require device inputs");
    // The legacy op's ring and shape rules; Q and K/V types follow the recipe (FAST takes prepared inputs).
    const uint32_t sq = q.logical_shape()[2];
    TT_FATAL(ring_size > 0 && ring_size % 2 == 0, "ring_size must be positive and even, got {}", ring_size);
    TT_FATAL(!ring_id || *ring_id < ring_size, "ring_id must be less than ring_size, got {}", ring_id.value_or(0));
    TT_FATAL(
        sq % (64 * ring_size) == 0,
        "Ring-distributed SDPA splits the sequence into 2 * ring_size tile-aligned slabs; got sequence length {} for "
        "ring size {}",
        sq,
        ring_size);
    TT_FATAL(
        chunk_start_idx.has_value() == page_table.has_value(),
        "Ring-distributed SDPA takes chunk_start_idx and page_table together (prefix caching)");
    TT_FATAL(
        page_table || k.logical_shape()[2] == sq,
        "Ring-distributed SDPA is causal and requires Q and K to have the same sequence length when not using prefix "
        "caching. Got Q: {}, K: {}",
        sq,
        k.logical_shape()[2]);
    const uint32_t slab_rows = q.logical_shape()[2] / (2 * ring_size);
    const auto slabs = [&](uint32_t id) {
        return std::array<uint32_t, 2>{id * slab_rows, (2 * ring_size - 1 - id) * slab_rows};
    };
    auto* mesh_device = q.device();
    numeric::RecipeKeyRange key_range{
        .causal = true,
        .q_offset = static_cast<uint32_t>(chunk_start_idx.value_or(0)),
        .page_table = page_table,
        .q_slab_rows = slab_rows};
    const ttnn::MeshCoordinateRange all_devices(mesh_device->shape());
    if (ring_id) {
        key_range.q_slab_starts.emplace_back(all_devices, slabs(*ring_id));
    } else {
        const auto& shape = mesh_device->get_view().shape();
        const bool rows = shape[0] == ring_size;
        TT_FATAL(
            rows || (shape.dims() > 1 && shape[1] == ring_size),
            "Ring size {} doesn't match mesh dimensions {}",
            ring_size,
            shape);
        for (const auto& coordinate : all_devices) {
            key_range.q_slab_starts.emplace_back(
                ttnn::MeshCoordinateRange(coordinate), slabs(coordinate[rows ? 0 : 1]));
        }
    }
    return dense_recipe(
        q, k, v, std::nullopt, scale, memory_config, program_config, compute_kernel_config, precision, key_range, {}, routed);
}
}  // namespace

ttnn::Tensor scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool is_causal,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& attention_sink,
    const std::optional<ttnn::Tensor>& cu_window_seqlens,
    uint32_t windowed_q_token_offset,
    const std::optional<ttnn::Tensor>& windowed_q_token_offset_tensor,
    bool output_concat_heads,
    std::optional<SDPAPrecision> precision) {
    if (!precision) {
        // Zero chunk sizes (op-chosen blocking) need an explicit recipe, routed or not.
        operations::transformer::sdpa::detail::reject_auto_blocking_without_recipe(program_config);
    }
    const bool routed = !precision && routes_fp32_dest(input_tensor_q, compute_kernel_config);
    if (precision || routed) {
        namespace numeric = operations::transformer::sdpa::detail;
        TT_FATAL(input_tensor_q.storage_type() == StorageType::DEVICE, "SDPA recipes require device inputs");
        // Causal, sliding-window and windowed calls run the K-range model (numeric::RecipeKeyRange), with the
        // legacy op's shape rules: causal and sliding windows need Sq == Sk; a windowed Q may be a slice of the
        // sequence starting at windowed_q_token_offset.
        const bool windowed = cu_window_seqlens.has_value();
        const uint32_t window = sliding_window_size.value_or(0);
        const uint32_t sq = input_tensor_q.logical_shape()[2], sk = input_tensor_k.logical_shape()[2];
        TT_FATAL(
            windowed || (windowed_q_token_offset == 0 && !windowed_q_token_offset_tensor),
            "windowed_q_token_offset requires cu_window_seqlens");
        TT_FATAL(
            !attn_mask || (!is_causal && window == 0 && !windowed),
            "SDPA recipes take either attn_mask or is_causal / sliding_window_size / cu_window_seqlens");
        if (windowed) {
            TT_FATAL(window == 0, "Windowed SDPA does not support sliding_window_size");
            TT_FATAL(!attention_sink, "Windowed SDPA does not support attention_sink");
            TT_FATAL(sq <= sk, "windowed Q shard has {} rows, more than the K sequence length {}", sq, sk);
            TT_FATAL(
                windowed_q_token_offset_tensor || windowed_q_token_offset <= sk - sq,
                "windowed Q shard [{}, {} + {}) does not fit in the K sequence length {}",
                windowed_q_token_offset,
                windowed_q_token_offset,
                sq,
                sk);
        } else if (is_causal || window > 0) {
            TT_FATAL(
                sq == sk,
                "Causal or sliding-window SDPA requires Q and K to have the same sequence length. Got Q: {}, K: {}",
                sq,
                sk);
        }
        const numeric::RecipeKeyRange key_range{
            .causal = is_causal,
            .sliding_window = window,
            .q_offset = windowed ? windowed_q_token_offset : 0,
            .q_offset_tensor = windowed_q_token_offset_tensor,
            .segments = cu_window_seqlens};
        return dense_recipe(
            input_tensor_q,
            input_tensor_k,
            input_tensor_v,
            attn_mask,
            scale,
            memory_config,
            program_config,
            compute_kernel_config,
            precision.value_or(SDPAPrecision::ACCURATE),
            key_range,
            {.attention_sink = attention_sink, .output_concat_heads = output_concat_heads},
            routed);
    }
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    // PyTorch semantics: softmax(Q·Kᵀ * scale + mask) · V, where `scale` applies
    // to Q·Kᵀ only and the mask is added unscaled.
    //
    // The compute kernel folds `scale` into the softmax exponent as a
    // performance optimization:
    //     exp((QK + mask - row_max) * scale)
    //   = exp(QK*scale + mask*scale - row_max*scale)
    // which scales the mask along with QK, diverging from PyTorch semantics.
    //
    // Pre-multiply the mask by 1/scale so the kernel's subsequent *scale
    // restores the original mask magnitude inside softmax. QK remains scaled
    // exactly once.
    //
    // Windowed mode synthesizes a {0, -inf} block-diagonal mask on-device from cu_window_seqlens;
    // pre-scaling is unnecessary (0/-inf are scale-invariant), so attn_mask is left empty.
    std::optional<ttnn::Tensor> effective_mask = attn_mask;
    if (attn_mask.has_value()) {
        const float effective_scale =
            scale.value_or(1.0f / std::sqrt(static_cast<float>(input_tensor_q.padded_shape()[-1])));
        if (effective_scale != 1.0f) {
            effective_mask = ttnn::multiply(attn_mask.value(), 1.0f / effective_scale);
        }
    }

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        effective_mask,
        std::nullopt,  // page_table
        attention_sink,
        is_causal,
        scale,
        sliding_window_size,
        std::nullopt,  // chunk_start_idx
        std::nullopt,  // chunk_start_idx_tensor
        false,         // use_mla
        std::nullopt,  // head_dim_v
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val,
        cu_window_seqlens,
        windowed_q_token_offset,
        windowed_q_token_offset_tensor,
        std::nullopt,
        output_concat_heads);
}

// Legacy: chunk_start_idx as scalar (part of program cache key).
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<ttnn::operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<ttnn::Tensor>& attention_sink,
    std::optional<SDPAPrecision> precision) {
    if (!precision) {
        operations::transformer::sdpa::detail::reject_auto_blocking_without_recipe(program_config);
    }
    const bool routed = !precision && routes_fp32_dest(input_tensor_q, compute_kernel_config);
    if (precision || routed) {
        return chunked_recipe(
            input_tensor_q,
            input_tensor_k,
            input_tensor_v,
            page_table_tensor,
            chunk_start_idx,
            std::nullopt,
            scale,
            memory_config,
            program_config,
            compute_kernel_config,
            paged_cache_geometry,
            sliding_window_size,
            attention_sink,
            precision.value_or(SDPAPrecision::ACCURATE),
            0,
            routed);
    }
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        std::nullopt,       // attn_mask
        page_table_tensor,  // page_table
        attention_sink,
        /*is_causal=*/true,  // Always causal for chunked version
        scale,
        sliding_window_size,
        chunk_start_idx,
        std::nullopt,  // chunk_start_idx_tensor
        false,         // use_mla
        std::nullopt,  // head_dim_v
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val,
        std::nullopt,  // cu_window_seqlens
        0,             // windowed_q_token_offset (windowed mode only)
        std::nullopt,  // windowed_q_token_offset_tensor
        paged_cache_geometry);
}

// Flexible: chunk_start_idx in device tensor [1]; read at runtime (for tracing).
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    const ttnn::Tensor& chunk_start_idx_tensor,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<ttnn::operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<ttnn::Tensor>& attention_sink,
    std::optional<SDPAPrecision> precision) {
    if (!precision) {
        operations::transformer::sdpa::detail::reject_auto_blocking_without_recipe(program_config);
    }
    const bool routed = !precision && routes_fp32_dest(input_tensor_q, compute_kernel_config);
    if (precision || routed) {
        return chunked_recipe(
            input_tensor_q,
            input_tensor_k,
            input_tensor_v,
            page_table_tensor,
            std::nullopt,
            chunk_start_idx_tensor,
            scale,
            memory_config,
            program_config,
            compute_kernel_config,
            paged_cache_geometry,
            sliding_window_size,
            attention_sink,
            precision.value_or(SDPAPrecision::ACCURATE),
            0,
            routed);
    }
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        std::nullopt,       // attn_mask
        page_table_tensor,  // page_table
        attention_sink,
        /*is_causal=*/true,
        scale,
        sliding_window_size,
        std::nullopt,
        chunk_start_idx_tensor,
        false,         // use_mla
        std::nullopt,  // head_dim_v
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val,
        std::nullopt,  // cu_window_seqlens
        0,             // windowed_q_token_offset (windowed mode only)
        std::nullopt,  // windowed_q_token_offset_tensor
        paged_cache_geometry);
}

std::tuple<ttnn::Tensor, ttnn::Tensor> joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& joint_tensor_q,
    const ttnn::Tensor& joint_tensor_k,
    const ttnn::Tensor& joint_tensor_v,
    const std::string& joint_strategy,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<SDPAPrecision> precision) {
    // Non-ring joint has no streaming kernel: without `precision` it always reaches the legacy loop, so it is
    // always routed (STANDARD, or ACCURATE with FP32 DEST).
    if (!precision) {
        operations::transformer::sdpa::detail::reject_auto_blocking_without_recipe(program_config);
    }
    const bool routed = !precision && routes_to_recipes(input_tensor_q);
    if (routed) {
        precision = routed_precision(input_tensor_q, compute_kernel_config);
    }
    if (precision) {
        namespace numeric = operations::transformer::sdpa::detail;
        TT_FATAL(joint_strategy == "rear", "SDPA recipes require rear joint strategy");
        const auto recipe_program_config =
            routed ? routed_program_config(program_config) : std::optional(program_config);
        const auto policy = numeric::resolve_recipe_policy(
            input_tensor_q, input_tensor_k, *precision, scale, compute_kernel_config, recipe_program_config);
        // BF16 kernel I/O; BFP8/BFP4 Q round-trip as in the dense branch.
        const auto query = numeric::recipe_bf16_query(input_tensor_q);
        const auto joint_query = numeric::recipe_bf16_query(joint_tensor_q);
        const auto blocking = numeric::resolve_dense_recipe_blocking(
            policy,
            query,
            input_tensor_k,
            &joint_query,
            &joint_tensor_k,
            recipe_program_config,
            nullptr,
            0,
            nullptr,
            nullptr,
            routed);
        auto [output, joint_output] = numeric::run_joint_recipe(
            query,
            input_tensor_k,
            input_tensor_v,
            joint_query,
            joint_tensor_k,
            joint_tensor_v,
            policy,
            blocking,
            scale);
        if (input_tensor_q.dtype() != DataType::BFLOAT16) {
            output = ttnn::typecast(output, input_tensor_q.dtype());
        }
        if (joint_tensor_q.dtype() != DataType::BFLOAT16) {
            joint_output = ttnn::typecast(joint_output, joint_tensor_q.dtype());
        }
        return {output, joint_output};
    }
    auto output_tensors = ttnn::prim::joint_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v,
        joint_strategy,
        program_config,
        scale,
        compute_kernel_config);
    return {output_tensors[prim::JOINT_SDPA_OUTPUT_IDX], output_tensors[prim::JOINT_SDPA_JOINT_OUTPUT_IDX]};
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ring_joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    const std::string& joint_strategy,
    const LogicalLength& logical_n,
    const LogicalLength& logical_l,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const CoreCoord ccl_core_grid_offset,
    bool is_causal,
    bool is_balanced,
    bool is_cross,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    std::optional<uint32_t> kv_cache_batch_idx,
    std::optional<uint32_t> kv_actual_isl,
    const std::optional<ttnn::Tensor>& attention_sink,
    std::optional<uint32_t> sliding_window_size,
    bool circular_kv_cache,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_joint_k,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_joint_v,
    const std::optional<ttnn::Tensor>& slot_id,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor,
    std::optional<uint32_t> kv_cache_num_layers,
    std::optional<uint32_t> kv_cache_layer_idx,
    std::optional<SDPAPrecision> precision) {
    if (!precision) {
        operations::transformer::sdpa::detail::reject_auto_blocking_without_recipe(program_config);
    }
    // Only FP32 DEST reaches the legacy loop (sdpa_ring); routed when the ring recipe has the call's features.
    bool routed = false;
    if (!precision && routes_fp32_dest(input_tensor_q, compute_kernel_config)) {
        if (const auto gap = ring_recipe_gap(
                input_tensor_q,
                input_tensor_k,
                input_tensor_v,
                is_cross,
                attention_sink,
                sliding_window_size,
                circular_kv_cache,
                kv_cache_batch_idx,
                kv_actual_isl,
                slot_id,
                kv_actual_isl_tensor)) {
            log_debug(
                tt::LogOp, "ring_joint SDPA with FP32 dest keeps the legacy loop: the ring recipe lacks {}", *gap);
        } else {
            routed = true;
            precision = SDPAPrecision::ACCURATE;
            program_config.sub_core_grids = std::nullopt;
        }
    }
    ttnn::Tensor query = input_tensor_q;
    std::optional<ttnn::Tensor> joint_query = joint_tensor_q;
    if (precision) {
        const auto policy = operations::transformer::sdpa::detail::resolve_recipe_policy(
            input_tensor_q, input_tensor_k, *precision, scale, compute_kernel_config, program_config);
        if (routed) {
            drop_unsupported_routed_chunks(
                program_config,
                operations::transformer::sdpa::detail::RecipeOp::Ring,
                policy,
                input_tensor_q.logical_shape()[3]);
        }
        TT_FATAL(
            !sliding_window_size,
            "Named ring recipes do not support sliding_window_size yet; omit precision for the legacy kernel");
        TT_FATAL(
            !attention_sink && !circular_kv_cache && !kv_cache_batch_idx && !kv_actual_isl && !slot_id &&
                !kv_actual_isl_tensor,
            "Named ring recipes do not support indexed/cache or sink features");
        const bool auto_q_chunk = program_config.q_chunk_size == 0;
        program_config = operations::transformer::sdpa::detail::resolve_ring_recipe_blocking(
            policy,
            input_tensor_q,
            input_tensor_k,
            joint_tensor_q,
            joint_tensor_k,
            static_cast<uint32_t>(
                cluster_axis == 0 ? mesh_device.get_view().num_rows() : mesh_device.get_view().num_cols()),
            program_config);
        TT_FATAL(
            input_tensor_k.logical_shape()[3] == input_tensor_q.logical_shape()[3] &&
                input_tensor_v.logical_shape()[3] == input_tensor_q.logical_shape()[3],
            "Named ring recipes require matching Q/K/V head dims");
        if (is_balanced && auto_q_chunk) {
            // A balanced ring's Q chunks must not straddle the two halves of a device's sequence.
            const uint32_t half_tiles = input_tensor_q.padded_shape()[2] / 64;
            uint32_t q_tiles = program_config.q_chunk_size / 32;
            while (q_tiles > 1 && half_tiles % q_tiles != 0) {
                --q_tiles;
            }
            program_config.q_chunk_size = q_tiles * 32;
        }
        // Unfused STANDARD processes Q tile rows in pairs: an odd Q chunk rounds up to the next even one, so that
        // kernel never builds the single-row group (outputs are per row, so the chunking does not change them
        // beyond accumulation order).
        program_config.q_chunk_size = operations::transformer::sdpa::detail::recipe_compute_q_tiles(
                                          policy, program_config.q_chunk_size / 32, program_config.k_chunk_size / 32) *
                                      32;
        operations::transformer::sdpa::detail::validate_recipe_geometry(
            operations::transformer::sdpa::detail::RecipeOp::Ring,
            policy,
            program_config.q_chunk_size,
            program_config.k_chunk_size,
            input_tensor_q.logical_shape()[3]);
        TT_FATAL(input_tensor_k.dtype() == input_tensor_v.dtype(), "Named ring recipes require matching KV types");
        TT_FATAL(
            is_cross || input_tensor_q.logical_shape()[2] == input_tensor_k.logical_shape()[2],
            "Named ring recipes do not yet support chunked prefill; use is_cross for noncausal cross attention");
        // The recipe fixes its exp: an exp_approx_mode=False is ignored (resolve_recipe_policy). Read only by the op
        // perf model; the recipe kernels fix their own fidelities, so a caller's compute config is replaced.
        program_config.exp_approx_mode = std::nullopt;
        compute_kernel_config = BlackholeComputeKernelConfig{
            .math_fidelity = policy.pv_fidelity,
            .math_approx_mode = true,
            .fp32_dest_acc_en = policy.fp32_destination,
        };
        // The ring recipe reads Q as BF16: a BFP8/BFP4 Q (legacy takes them) is widened first and the outputs
        // narrowed back, as on the dense path.
        query = operations::transformer::sdpa::detail::recipe_bf16_query(input_tensor_q);
        if (joint_tensor_q && joint_tensor_q->logical_shape().volume() > 0) {
            joint_query = operations::transformer::sdpa::detail::recipe_bf16_query(*joint_tensor_q);
        }
    }
    // Normalize empty joints to nullopt (see drop_if_empty).
    const std::optional<ttnn::Tensor> joint_q = drop_if_empty(joint_query);
    const std::optional<ttnn::Tensor> joint_k = drop_if_empty(joint_tensor_k);
    const std::optional<ttnn::Tensor> joint_v = drop_if_empty(joint_tensor_v);

    // Split each logical length into (scalar attribute, optional device tensor); on the tensor path the
    // attribute becomes the worst-case placeholder (see RingJointSDPAInputs).
    const std::size_t ring_size =
        (cluster_axis == 0) ? mesh_device.get_view().num_rows() : mesh_device.get_view().num_cols();
    const std::size_t padded_ring_n = static_cast<std::size_t>(input_tensor_k.logical_shape()[2]) * ring_size;
    const std::size_t padded_ring_l =
        joint_k.has_value() ? static_cast<std::size_t>(joint_k->logical_shape()[2]) * ring_size : 0;
    const auto split_logical_length =
        [](const LogicalLength& length,
           std::size_t placeholder) -> std::pair<std::size_t, std::optional<ttnn::Tensor>> {
        if (const auto* scalar = std::get_if<std::size_t>(&length)) {
            return {*scalar, std::nullopt};
        }
        return {placeholder, std::get<ttnn::Tensor>(length)};
    };
    const auto [logical_n_scalar, logical_n_tensor] = split_logical_length(logical_n, padded_ring_n);
    const auto [logical_l_scalar, logical_l_tensor] = split_logical_length(logical_l, padded_ring_l);

    auto topology_1d = ttnn::ccl::convert_2d_to_1d_topology(topology);
    auto output_tensors = ttnn::prim::ring_joint_scaled_dot_product_attention(
        query,
        input_tensor_k,  // AllGather input
        input_tensor_v,  // AllGather input
        joint_q,
        joint_k,
        joint_v,
        persistent_output_buffer_k,  // AllGather output / RingAttention input
        persistent_output_buffer_v,  // AllGather output / RingAttention input
        persistent_output_buffer_joint_k,
        persistent_output_buffer_joint_v,
        joint_strategy,
        logical_n_scalar,
        logical_l_scalar,
        std::move(program_config),
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        topology_1d,
        ccl_core_grid_offset,
        subdevice_id,
        is_causal,
        is_balanced,
        is_cross,
        scale,
        compute_kernel_config,
        core_allocation_strategy,
        kv_cache_batch_idx,
        kv_actual_isl,
        std::nullopt,  // latent_v_head_dim
        attention_sink,
        slot_id,
        kv_actual_isl_tensor,
        // Resolve to (1, 0) when unset so the readers compute slot = slot_id[0], the
        // pre-existing behaviour for callers that pass no layer packing.
        kv_cache_num_layers.value_or(1),
        kv_cache_layer_idx.value_or(0),
        sliding_window_size,
        circular_kv_cache,
        logical_n_tensor,
        logical_l_tensor,
        precision);
    auto& output = output_tensors[prim::RING_JOINT_SDPA_OUTPUT_IDX];
    auto& joint_output = output_tensors[prim::RING_JOINT_SDPA_JOINT_OUTPUT_IDX];
    if (output.dtype() != input_tensor_q.dtype()) {
        output = ttnn::typecast(output, input_tensor_q.dtype());
    }
    if (joint_tensor_q && joint_output.dtype() != joint_tensor_q->dtype() &&
        joint_output.logical_shape().volume() > 0) {
        joint_output = ttnn::typecast(joint_output, joint_tensor_q->dtype());
    }
    return {output, joint_output, output_tensors[prim::RING_JOINT_SDPA_STATS_OUTPUT_IDX]};
}

std::tuple<ttnn::Tensor, ttnn::Tensor> ring_mla(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_kv,
    ttnn::Tensor& persistent_output_buffer_kv,
    const uint32_t head_dim_v,
    std::size_t logical_n,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<uint32_t> cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const CoreCoord ccl_core_grid_offset,
    bool is_balanced,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    std::optional<uint32_t> kv_cache_batch_idx,
    std::optional<uint32_t> kv_actual_isl,
    const std::optional<ttnn::Tensor>& slot_id,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor,
    std::optional<uint32_t> kv_cache_num_layers,
    std::optional<uint32_t> kv_cache_layer_idx) {
    auto output_tensors = ttnn::prim::ring_joint_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_kv,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        persistent_output_buffer_kv,
        std::nullopt,  // persistent_output_buffer_v
        std::nullopt,  // persistent_output_buffer_joint_k
        std::nullopt,  // persistent_output_buffer_joint_v
        "rear",
        logical_n,
        /*logical_l=*/static_cast<std::size_t>(0),
        std::move(program_config),
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        topology,
        ccl_core_grid_offset,
        subdevice_id,
        /*is_causal=*/true,
        is_balanced,
        /*is_cross=*/false,
        scale,
        compute_kernel_config,
        core_allocation_strategy,
        kv_cache_batch_idx,
        kv_actual_isl,
        head_dim_v,
        std::nullopt,  // attention_sink
        slot_id,
        kv_actual_isl_tensor,
        kv_cache_num_layers.value_or(1),
        kv_cache_layer_idx.value_or(0),
        std::nullopt);  // sliding_window_size
    return {output_tensors[prim::RING_JOINT_SDPA_OUTPUT_IDX], output_tensors[prim::RING_JOINT_SDPA_STATS_OUTPUT_IDX]};
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ExecuteExpRingJointAttention::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    const std::string& joint_strategy,
    const LogicalLength& logical_n,
    operations::transformer::SDPAProgramConfig program_config,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const uint32_t num_workers_per_link,
    const uint32_t num_buffers_per_channel,
    std::optional<SDPAPrecision> precision) {
    // The legacy exp ring kernel builds only its streaming path; any other blocking (and FP32 DEST) fails to
    // compile there, so those calls run a recipe (STANDARD, or ACCURATE with FP32 DEST).
    if (!precision) {
        operations::transformer::sdpa::detail::reject_auto_blocking_without_recipe(program_config);
    }
    bool routed = false;
    if (!precision && routes_to_recipes(input_tensor_q)) {
        const auto legacy_config = legacy_compute_config(input_tensor_q, compute_kernel_config);
        routed = !ttnn::prim::detail::exp_ring_streaming_compute_supported(
            program_config.q_chunk_size / 32,
            program_config.k_chunk_size / 32,
            ttnn::get_dest_reg_count(legacy_config),
            get_fp32_dest_acc_en(legacy_config));
        if (routed) {
            precision = routed_precision(input_tensor_q, compute_kernel_config);
            program_config.sub_core_grids = std::nullopt;
        }
    }
    if (precision) {
        // The recipe owns the numerics: resolve_recipe_policy accepts and ignores compute_kernel_config and
        // exp_approx_mode=False, and validates the scale.
        const auto policy = operations::transformer::sdpa::detail::resolve_recipe_policy(
            input_tensor_q, input_tensor_k, *precision, scale, compute_kernel_config, program_config);
        if (routed) {
            drop_unsupported_routed_chunks(
                program_config,
                operations::transformer::sdpa::detail::RecipeOp::ExpRing,
                policy,
                input_tensor_q.logical_shape()[3]);
        }
        program_config = operations::transformer::sdpa::detail::resolve_exp_ring_recipe_blocking(
            policy,
            input_tensor_q,
            input_tensor_k,
            joint_tensor_q,
            static_cast<uint32_t>(
                cluster_axis == 0 ? mesh_device.get_view().num_rows() : mesh_device.get_view().num_cols()),
            program_config);
        TT_FATAL(
            input_tensor_k.logical_shape()[3] == input_tensor_q.logical_shape()[3] &&
                input_tensor_v.logical_shape()[3] == input_tensor_q.logical_shape()[3],
            "Named exp ring recipes require matching Q/K/V head dims");
        // Unfused STANDARD processes Q tile rows in pairs: an odd Q chunk rounds up to the next even one, so that
        // kernel never builds the single-row group (outputs are per row, so the chunking does not change them
        // beyond accumulation order).
        program_config.q_chunk_size = operations::transformer::sdpa::detail::recipe_compute_q_tiles(
                                          policy, program_config.q_chunk_size / 32, program_config.k_chunk_size / 32) *
                                      32;
        operations::transformer::sdpa::detail::validate_recipe_geometry(
            operations::transformer::sdpa::detail::RecipeOp::ExpRing,
            policy,
            program_config.q_chunk_size,
            program_config.k_chunk_size,
            input_tensor_q.logical_shape()[3]);
        TT_FATAL(
            input_tensor_q.dtype() == DataType::BFLOAT16 && input_tensor_k.dtype() == input_tensor_v.dtype(),
            "Named exp ring recipes require BF16 Q and matching KV types");
        // The recipe fixes its exp: an exp_approx_mode=False is ignored (resolve_recipe_policy). Read only by the op
        // perf model; the recipe kernels fix their own fidelities, so a caller's compute config is replaced.
        program_config.exp_approx_mode = std::nullopt;
        compute_kernel_config = BlackholeComputeKernelConfig{
            .math_fidelity = policy.pv_fidelity,
            .math_approx_mode = true,
            .fp32_dest_acc_en = policy.fp32_destination,
        };
    }
    // Normalize empty joints to nullopt (see drop_if_empty).
    const std::optional<ttnn::Tensor> joint_q = drop_if_empty(joint_tensor_q);
    const std::optional<ttnn::Tensor> joint_k = drop_if_empty(joint_tensor_k);
    const std::optional<ttnn::Tensor> joint_v = drop_if_empty(joint_tensor_v);

    // Tensor path: the scalar attribute becomes the worst-case placeholder (see ExpRingJointSDPAInputs).
    const std::size_t ring_size =
        (cluster_axis == 0) ? mesh_device.get_view().num_rows() : mesh_device.get_view().num_cols();
    const std::size_t padded_ring_n = static_cast<std::size_t>(input_tensor_k.logical_shape()[2]) * ring_size;
    std::size_t logical_n_scalar = padded_ring_n;
    std::optional<ttnn::Tensor> logical_n_tensor;
    if (const auto* scalar = std::get_if<std::size_t>(&logical_n)) {
        logical_n_scalar = *scalar;
    } else {
        logical_n_tensor = std::get<ttnn::Tensor>(logical_n);
    }

    auto output_tensors = ttnn::prim::exp_ring_joint_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_k,  // AllGather input
        input_tensor_v,  // AllGather input
        joint_q,
        joint_k,
        joint_v,
        persistent_output_buffer_k,  // AllGather output / RingAttention input
        persistent_output_buffer_v,  // AllGather output / RingAttention input
        joint_strategy,
        logical_n_scalar,
        std::move(program_config),
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        topology,
        subdevice_id,
        scale,
        compute_kernel_config,
        num_workers_per_link,
        num_buffers_per_channel,
        logical_n_tensor,
        precision);
    return {
        output_tensors[prim::EXP_RING_JOINT_SDPA_OUTPUT_IDX],
        output_tensors[prim::EXP_RING_JOINT_SDPA_JOINT_OUTPUT_IDX],
        output_tensors[prim::EXP_RING_JOINT_SDPA_STATS_OUTPUT_IDX]};
}

ttnn::Tensor flash_mla_prefill(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const std::optional<ttnn::Tensor>& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool is_causal,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<SDPAPrecision> precision) {
    if (!precision) {
        operations::transformer::sdpa::detail::reject_auto_blocking_without_recipe(program_config);
    }
    const bool routed = !precision && routes_fp32_dest(input_tensor_q, compute_kernel_config);
    if (precision || routed) {
        // V is K's first head_dim_v columns unless given. Causal needs Sq == Sk, as for legacy SDPA.
        TT_FATAL(input_tensor_q.storage_type() == StorageType::DEVICE, "SDPA recipes require device inputs");
        TT_FATAL(
            !attn_mask || !is_causal,
            "SDPA recipes take either attn_mask or is_causal, got both for flash_mla_prefill");
        TT_FATAL(
            !is_causal || input_tensor_q.logical_shape()[2] == input_tensor_k.logical_shape()[2],
            "Causal MLA prefill requires Q and K to have the same sequence length. Got Q: {}, K: {}",
            input_tensor_q.logical_shape()[2],
            input_tensor_k.logical_shape()[2]);
        return dense_recipe(
            input_tensor_q,
            input_tensor_k,
            input_tensor_v.value_or(input_tensor_k),
            attn_mask,
            scale,
            memory_config,
            program_config,
            compute_kernel_config,
            precision.value_or(SDPAPrecision::ACCURATE),
            {.causal = is_causal},
            {.head_dim_v = head_dim_v},
            routed);
    }
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        attn_mask,
        std::nullopt,  // page_table
        std::nullopt,  // attention_sink
        is_causal,
        scale,
        std::nullopt,  // sliding_window_size (not supported yet)
        std::nullopt,  // chunk_start_idx
        std::nullopt,  // chunk_start_idx_tensor
        true,          // use_mla
        head_dim_v,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val);
}

ttnn::Tensor chunked_flash_mla_prefill(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<SDPAPrecision> precision) {
    if (!precision) {
        operations::transformer::sdpa::detail::reject_auto_blocking_without_recipe(program_config);
    }
    const bool routed = !precision && routes_fp32_dest(input_tensor_q, compute_kernel_config);
    if (precision || routed) {
        return chunked_recipe(
            input_tensor_q,
            input_tensor_k,
            input_tensor_k,  // V is K's first head_dim_v columns
            page_table_tensor,
            chunk_start_idx,
            std::nullopt,
            scale,
            memory_config,
            program_config,
            compute_kernel_config,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            precision.value_or(SDPAPrecision::ACCURATE),
            head_dim_v,
            routed);
    }
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa(
        input_tensor_q,
        input_tensor_k,
        std::nullopt,       // V is implied by K in MLA mode
        std::nullopt,       // attn_mask
        page_table_tensor,  // page_table
        std::nullopt,       // attention_sink
        /*is_causal=*/true,
        scale,
        std::nullopt,  // sliding_window_size (not supported yet)
        chunk_start_idx,
        std::nullopt,  // chunk_start_idx_tensor
        true,          // use_mla
        head_dim_v,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val);
}

ttnn::Tensor ring_distributed_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    uint32_t ring_size,
    std::optional<uint32_t>
        ring_id,  // Optional: if provided, uses this value; if nullopt, infers from device coordinate
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& page_table,
    std::optional<int64_t> chunk_start_idx,
    std::optional<SDPAPrecision> precision) {
    // Without precision, Blackhole runs STANDARD, or ACCURATE with FP32 DEST (this op always left the streaming
    // kernels for the legacy loop).
    const bool routed = !precision && routes_to_recipes(input_tensor_q);
    if (routed) {
        precision = routed_precision(input_tensor_q, compute_kernel_config);
    }
    if (precision) {
        return ring_distributed_recipe(
            input_tensor_q,
            input_tensor_k,
            input_tensor_v,
            ring_size,
            ring_id,
            scale,
            memory_config,
            routed ? routed_program_config(program_config) : program_config,
            compute_kernel_config,
            page_table,
            chunk_start_idx,
            *precision,
            routed);
    }
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::ring_distributed_sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        ring_size,
        ring_id,  // Pass through the ring_id parameter (can be used or ignored)
        scale,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        program_config,
        kernel_config_val,
        page_table,
        chunk_start_idx);
}

}  // namespace ttnn::transformer
