// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <tuple>
#include <utility>

#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_fusion.hpp"

namespace ttnn::prim {

struct RingJointSDPAParams {
    std::string joint_strategy;
    std::optional<float> scale;
    bool is_causal = false;
    bool is_balanced = false;
    bool is_cross = false;
    std::size_t logical_n = 0;
    std::size_t logical_l = 0;
    std::size_t ring_size = 0;
    tt::tt_metal::MemoryConfig output_memory_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    experimental::prim::RingAttentionAllGatherAsyncParams all_gather_operation_attributes;
    experimental::prim::RingAttentionAllGatherAsyncInputs all_gather_tensor_args;
    CoreCoord ccl_core_grid_offset;
    std::optional<std::uint32_t> kv_cache_batch_idx = std::nullopt;
    std::optional<std::uint32_t> kv_actual_isl = std::nullopt;
    uint32_t latent_v_head_dim = 0;
    uint32_t kv_cache_num_layers = 1;
    uint32_t kv_cache_layer_idx = 0;
    std::optional<uint32_t> sliding_window_size = std::nullopt;
    // Circular (bounded) sliding KV cache: the local K/V shard is a circular buffer of whole
    // Q-sized slabs (chunk group g lives in local slab g % n_slabs; the writer wraps host-side).
    // The slab count is DERIVED on-device from the cache/Q geometry (N_local_kv / N_local_q) —
    // callers only opt in. false = unbounded cache (byte-identical to the pre-existing behavior).
    // Requires chunked sliding + kv_actual_isl.
    bool circular_kv_cache = false;
    // Internal structural ratio, derived from Q/KV sequence distributions.
    uint32_t kv_stripe_split = 1;
    // Sol-Attn exact-set granularity ("auto" = infer from the mask width: nkc_pad -> chunk, else gather; or an
    // explicit "gather" | "chunk" | "compact" | "pack" | "resident"). Set after construction; hashed; see
    // RingJointSDPAInputs::sol_*_mode.
    std::string sol_mode = "auto";

    // We need a constructor, because all_gather_struct is not default initializable.
    RingJointSDPAParams(
        std::string joint_strategy,
        std::optional<float> scale,
        bool is_causal,
        bool is_balanced,
        bool is_cross,
        std::size_t logical_n,
        std::size_t logical_l,
        std::size_t ring_size,
        tt::tt_metal::MemoryConfig output_memory_config,
        std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
        DeviceComputeKernelConfig compute_kernel_config,
        experimental::prim::RingAttentionAllGatherAsyncParams all_gather_operation_attributes,
        experimental::prim::RingAttentionAllGatherAsyncInputs all_gather_tensor_args,
        CoreCoord ccl_core_grid_offset,
        std::optional<std::uint32_t> kv_cache_batch_idx = std::nullopt,
        std::optional<std::uint32_t> kv_actual_isl = std::nullopt,
        uint32_t latent_v_head_dim = 0,
        uint32_t kv_cache_num_layers = 1,
        uint32_t kv_cache_layer_idx = 0,
        std::optional<uint32_t> sliding_window_size = std::nullopt,
        bool circular_kv_cache = false,
        uint32_t kv_stripe_split = 1) :
        joint_strategy(std::move(joint_strategy)),
        scale(scale),
        is_causal(is_causal),
        is_balanced(is_balanced),
        is_cross(is_cross),
        logical_n(logical_n),
        logical_l(logical_l),
        ring_size(ring_size),
        output_memory_config(std::move(output_memory_config)),
        program_config(std::move(program_config)),
        compute_kernel_config(compute_kernel_config),
        all_gather_operation_attributes(std::move(all_gather_operation_attributes)),
        all_gather_tensor_args(std::move(all_gather_tensor_args)),
        ccl_core_grid_offset(ccl_core_grid_offset),
        kv_cache_batch_idx(kv_cache_batch_idx),
        kv_actual_isl(kv_actual_isl),
        latent_v_head_dim(latent_v_head_dim),
        kv_cache_num_layers(kv_cache_num_layers),
        kv_cache_layer_idx(kv_cache_layer_idx),
        sliding_window_size(sliding_window_size),
        circular_kv_cache(circular_kv_cache),
        kv_stripe_split(kv_stripe_split) {}

    std::uint32_t get_q_chunk_size() const { return program_config.has_value() ? program_config->q_chunk_size : 32; }

    std::uint32_t get_k_chunk_size() const { return program_config.has_value() ? program_config->k_chunk_size : 32; }

    bool has_indexed_kv_cache() const { return kv_cache_batch_idx.has_value(); }

    // Host-path cache batch: the same (user, layer) fold the readers apply to slot_id[0]. Identity with the defaults.
    std::optional<std::uint32_t> cache_batch_idx() const {
        return kv_cache_batch_idx.has_value()
                   ? std::optional<std::uint32_t>(*kv_cache_batch_idx * kv_cache_num_layers + kv_cache_layer_idx)
                   : std::nullopt;
    }

    bool has_kv_pad_rotation() const { return kv_actual_isl.has_value(); }

    uint32_t q_ring_size() const { return static_cast<uint32_t>(ring_size) / kv_stripe_split; }

    bool has_sliding_window() const { return sliding_window_size.value_or(0) > 0; }

    static constexpr auto attribute_names = std::forward_as_tuple(
        "joint_strategy",
        "scale",
        "is_causal",
        "is_balanced",
        "is_cross",
        "logical_n",
        "logical_l",
        "ring_size",
        "compute_kernel_config",
        "program_config",
        "ccl_core_grid_offset",
        "has_kv_cache_batch_idx",
        "host_kv_pad_rotation",
        "latent_v_head_dim",
        "sliding_window_size",
        "circular_kv_cache",
        "kv_stripe_split",
        "all_gather_operation_attributes",
        "all_gather_tensor_args",
        "sol_mode");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(joint_strategy),
            std::cref(scale),
            std::cref(is_causal),
            std::cref(is_balanced),
            std::cref(is_cross),
            // Reflection only; the program hash is computed explicitly (kv_pad_rotation_active needs the tensors).
            std::cref(logical_n),
            std::cref(logical_l),
            std::cref(ring_size),
            std::cref(compute_kernel_config),
            std::cref(program_config),
            std::cref(ccl_core_grid_offset),
            kv_cache_batch_idx.has_value(),
            has_kv_pad_rotation(),
            std::cref(latent_v_head_dim),
            std::cref(sliding_window_size),
            std::cref(circular_kv_cache),
            std::cref(kv_stripe_split),
            std::cref(all_gather_operation_attributes),
            std::cref(all_gather_tensor_args),
            std::cref(sol_mode));
    }
};

struct RingJointSDPAInputs {
    Tensor input_q;
    Tensor input_k;
    std::optional<Tensor> input_v;
    std::optional<Tensor> joint_q;
    std::optional<Tensor> joint_k;
    std::optional<Tensor> joint_v;
    Tensor gathered_k;
    std::optional<Tensor> gathered_v;
    std::optional<Tensor> attention_sink;

    // Populated on the sharded-joint path only; nullopt on the replicated path.
    // Invariant: joint_is_sharded() <=> gathered_joint_k.has_value()
    std::optional<Tensor> gathered_joint_k;
    std::optional<Tensor> gathered_joint_v;

    // Trace-safe metadata path (both or neither): 1-element uint32 DRAM tensors the kernels read at start, slot_id[0]
    // the cache-user slot and kv_actual_isl[0] the prior valid global KV length, so one captured program replays
    // across chunks and users. Contract: ring_joint_scaled_dot_product_attention docstring (sdpa_nanobind.cpp).
    std::optional<Tensor> slot_id;
    std::optional<Tensor> kv_actual_isl;

    // Trace-safe transport for logical_n / logical_l (opt-in, independent of the metadata path above):
    // single-valued uint32/int32 device tensors read on-device each dispatch. When set, the matching
    // RingJointSDPAParams scalar is a worst-case capacity placeholder (the padded ring total), so every
    // host-side derivation stays valid and the kernels narrow it per dispatch.
    std::optional<Tensor> logical_n_tensor;
    std::optional<Tensor> logical_l_tensor;

    // Sol-Attn block-sparse mode. Contract: models/tt_dit/models/transformers/minimax_h3/SOL_ATTN_INTERFACE.md.
    // The routing pre-pass produces the first three; mask + pooled K + pooled V together, or none:
    //   sol_block_mask  uint32 row-major [1, NH, nqc_pad, W]. Chunk mode (W = nkc_pad, global K chunks): 1 =
    //                   compute this (local Q chunk, global K chunk) exactly. Gather mode (W = n_blocks_pad,
    //                   global 64-token blocks): 1 = gather this block into the Q chunk's exact K/V chunks.
    //   sol_pooled_k/v  bf16 tile [1, NH, n_blocks_pad, DH]: per-64-token-block mean K / mean V over the valid
    //                   tokens, whole sequence, per-shard blocking (shard s at rows [s*nb_pad, (s+1)*nb_pad)).
    //   sol_bias        OPTIONAL bf16 tile [1, NH, nqc_pad * 32, n_blocks_pad]: per Q chunk one tile row of
    //                   column bias for the pooled logits (ln(len)/scale for approximated blocks, a large
    //                   negative for exact and padding blocks). When absent the reader generates the same
    //                   tile rows on device from the mask row and the block geometry (sol_bias_gen.hpp); a
    //                   supplied tensor is used as-is instead.
    //   sol_row_max     Optional bf16 tile [1, NH, q_rows_pad, 32] (q_rows_pad = the local Q shard's padded row
    //                   count); accepted by sol_mode="resident" only and shape-validated, but ignored: resident
    //                   compute always runs its own (lazy) running row max. Omit it.
    std::optional<Tensor> sol_block_mask;
    std::optional<Tensor> sol_pooled_k;
    std::optional<Tensor> sol_pooled_v;
    std::optional<Tensor> sol_bias;
    std::optional<Tensor> sol_row_max;

    bool sol_enabled() const { return sol_block_mask.has_value(); }
    bool sol_bias_generated() const { return sol_enabled() && !sol_bias.has_value(); }
    // sol_row_max supplied (resident only). Only selects the reader's runtime-arg layout; compute ignores it.
    bool sol_static_max() const { return sol_enabled() && sol_row_max.has_value(); }
    // Exact-set granularity. "chunk": the mask is [1, NH, nqc_pad, nkc_pad] over K CHUNKS; the reader streams every
    // chunk through the chained path (minus chain-wide skips) and compute drains the chunks the mask drops.
    // "gather": block mask [1, NH, nqc_pad, n_blocks_pad]; the reader gathers kept blocks from DRAM (chains off).
    // "compact": block mask too, but streamed through the chains like chunk mode; each core left-justifies its own
    // kept blocks inside the chunk page and compute runs a narrowed step.
    // "auto" infers chunk vs gather from the mask width.
    bool sol_mask_is_chunk_width() const {
        return sol_enabled() && sol_block_mask->logical_shape()[3] != sol_pooled_k->logical_shape()[2];
    }
    // "resident": the K/V-once loop order: every K/V piece is streamed through a core once per ring iteration
    // (chains on, as chunk mode) and applied to each of the core's resident Q chunks, whose accumulators stay in
    // L1 for the whole op. Its exact set is per 64-token BLOCK: the mask is the block-form one (W = n_blocks_pad,
    // as compact), a piece is streamed when any chunk of the chain keeps a block of it, and each chunk's step
    // multiplies only its kept blocks in place. Requires the generated bias.
    bool sol_resident_mode(const std::string& mode) const { return sol_enabled() && mode == "resident"; }
    bool sol_chunk_mode(const std::string& mode) const {
        return sol_enabled() &&
               (mode == "chunk" || sol_resident_mode(mode) || (mode == "auto" && sol_mask_is_chunk_width()));
    }
    // "pack": compact plus cross-chunk packing (kept blocks appended into full pages). A mode rather than an
    // env var so it participates in the program hash and the kernels' named compile-time args.
    bool sol_pack_mode(const std::string& mode) const { return sol_enabled() && mode == "pack"; }
    bool sol_compact_mode(const std::string& mode) const {
        return sol_enabled() && (mode == "compact" || sol_pack_mode(mode));
    }
    bool sol_gather_mode(const std::string& mode) const {
        return sol_enabled() && !sol_chunk_mode(mode) && !sol_compact_mode(mode);
    }
    // Streaming modes keep the K/V chains and the reader's dense chunk loop (chunk and compact).
    bool sol_streams_chunks(const std::string& mode) const { return sol_chunk_mode(mode) || sol_compact_mode(mode); }
    // Global real-token count (metadata path only); attention stops at the tile holding the last real token.
    std::optional<Tensor> kv_valid_end;

    bool has_metadata() const { return slot_id.has_value() && kv_actual_isl.has_value(); }

    bool has_logical_n_tensor() const { return logical_n_tensor.has_value(); }

    bool has_logical_l_tensor() const { return logical_l_tensor.has_value(); }

    bool has_kv_valid_end() const { return kv_valid_end.has_value(); }

    // Chunked-prefill is signalled implicitly by Q being shorter than the per-device K shard:
    // Q is the latest slab, K is the populated prefix from chunk 0 through the current chunk.
    uint32_t local_kv_seq_len() const { return static_cast<uint32_t>(input_k.logical_shape()[2]); }

    // Split KV is block-cyclic from depth one, even when local K is shorter than Q.
    bool is_chunked(uint32_t kv_stripe_split = 1) const {
        return kv_stripe_split > 1 || input_q.logical_shape()[2] < local_kv_seq_len();
    }

    // The metadata path derives KV-pad rotation on-device only for chunked prefill.
    bool kv_pad_from_metadata(uint32_t kv_stripe_split = 1) const {
        return has_metadata() && is_chunked(kv_stripe_split);
    }

    // Circular sliding KV: Q-sized chunk slabs per device in the K/V cache (validation requires
    // whole slabs, and at least two of them, before this is read).
    uint32_t kv_slab_count() const { return local_kv_seq_len() / static_cast<uint32_t>(input_q.logical_shape()[2]); }

    // Latent V: V is omitted and read from K's rows (a prefix of K, or its last vDHt columns when packed).
    bool has_latent_v() const { return !input_v.has_value(); }

    // Packed latent V: K rows are wider than Q's head dim; QK reads K's first DH columns and V its last VDH columns.
    bool has_packed_kv() const { return has_latent_v() && input_k.logical_shape()[3] > input_q.logical_shape()[3]; }

    // Latent V copied from the K chunk in L1.
    bool v_shares_k_buffer() const { return has_latent_v() && !has_packed_kv(); }

    uint32_t v_num_heads() const {
        return input_v.has_value() ? static_cast<uint32_t>(input_v->logical_shape()[1])
                                   : static_cast<uint32_t>(input_k.logical_shape()[1]);
    }

    uint32_t v_head_dim(uint32_t latent_v_head_dim) const {
        return input_v.has_value() ? static_cast<uint32_t>(input_v->logical_shape()[3]) : latent_v_head_dim;
    }

    // True iff the joint/prompt set is sequence-sharded (L/P per device).
    // Derived from buffer presence so the flag and the buffer cannot drift out of sync.
    bool joint_is_sharded() const { return gathered_joint_k.has_value(); }
};

// KV-pad rotation: the host passes kv_actual_isl, or the chunked metadata path derives it on-device. Program hash,
// compile-time zeroing and validation all go through this rule so they cannot disagree. The all-gather bound
// (compute_gather_valid_Ht) is the deliberate exception: it needs a host length, so it is host-path only.
inline bool kv_pad_rotation_active(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    return args.has_kv_pad_rotation() || tensor_args.kv_pad_from_metadata(args.kv_stripe_split);
}

// Single-slot indexed KV cache: the host passes kv_cache_batch_idx, or the metadata path supplies the slot on-device.
inline bool indexed_kv_cache_active(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    return args.has_indexed_kv_cache() || tensor_args.has_metadata();
}

// Index constants for RingJointSDPAResult vector
constexpr size_t RING_JOINT_SDPA_OUTPUT_IDX = 0;
constexpr size_t RING_JOINT_SDPA_JOINT_OUTPUT_IDX = 1;
constexpr size_t RING_JOINT_SDPA_STATS_OUTPUT_IDX = 2;

// RingJointSDPAResult is a vector of 3 tensors: [output, joint_output, stats_output]
using RingJointSDPAResult = Tensors;

// RingJointSDPAResultSpec is a vector of 3 TensorSpecs: [output, joint_output, stats_output]
using RingJointSDPAResultSpec = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttnn::prim
