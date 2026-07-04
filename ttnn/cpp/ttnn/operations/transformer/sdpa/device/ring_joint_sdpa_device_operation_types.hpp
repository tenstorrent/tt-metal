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
    // (user, layer)-major KV-cache batch dim (metadata path). The SDPA + all-gather readers compute the
    // cache slot on-device as metadata[0] * kv_cache_num_layers + kv_cache_layer_idx (mirrors
    // update_padded_kv_cache). Defaults (1, 0) reduce to metadata[0], so existing callers are unaffected.
    // Hashed (unlike the per-dispatch re-patched, un-hashed kv_cache_batch_idx): baked into the readers'
    // create-time args, so each (num_layers, layer_idx) keys its own per-layer program.
    uint32_t kv_cache_num_layers = 1;
    uint32_t kv_cache_layer_idx = 0;
    // Whether logical_n is derived on-device (rotation) rather than baked into the kernel. True for the scalar
    // kv_actual_isl path OR the trace-safe metadata path on chunked shapes -- i.e. has_kv_pad_rotation() ||
    // (metadata present && is_chunked). Set at op creation because it needs is_chunked (a tensor-shape
    // property the attributes can't see), mirroring program_factory's kv_pad_rotation_enabled. The program
    // hash keys off this so logical_n is excluded from the hash exactly when it's derived (one program reused
    // across all rotation chunks -- trace-safe) and included when it's baked.
    bool kv_pad_rotation_enabled = false;

    // We need a constructor, because all_gather_struct is not default initializable.
    RingJointSDPAParams(
        std::string joint_strategy,
        std::optional<float> scale,
        bool is_causal,
        bool is_balanced,
        bool is_cross,
        std::size_t logical_n,
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
        bool kv_pad_rotation_enabled = false) :
        joint_strategy(std::move(joint_strategy)),
        scale(scale),
        is_causal(is_causal),
        is_balanced(is_balanced),
        is_cross(is_cross),
        logical_n(logical_n),
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
        kv_pad_rotation_enabled(kv_pad_rotation_enabled) {}

    std::uint32_t get_q_chunk_size() const { return program_config.has_value() ? program_config->q_chunk_size : 32; }

    std::uint32_t get_k_chunk_size() const { return program_config.has_value() ? program_config->k_chunk_size : 32; }

    bool has_indexed_kv_cache() const { return kv_cache_batch_idx.has_value(); }

    bool has_kv_pad_rotation() const { return kv_actual_isl.has_value(); }

    static constexpr auto attribute_names = std::forward_as_tuple(
        "joint_strategy",
        "scale",
        "is_causal",
        "is_balanced",
        "is_cross",
        "cache_key_logical_n",
        "ring_size",
        "compute_kernel_config",
        "program_config",
        "ccl_core_grid_offset",
        "has_kv_cache_batch_idx",
        "kv_pad_rotation_enabled",
        "latent_v_head_dim",
        // (num_layers, layer_idx) are baked into the readers' create-time args (not re-patched), so each
        // pair must key a distinct program; defaults (1, 0) leave the hash unchanged for single-layer callers.
        "kv_cache_num_layers",
        "kv_cache_layer_idx",
        "all_gather_operation_attributes",
        "all_gather_tensor_args");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(joint_strategy),
            std::cref(scale),
            std::cref(is_causal),
            std::cref(is_balanced),
            std::cref(is_cross),
            // logical_n is excluded from the hash exactly when it's derived on-device (rotation: scalar OR
            // metadata) -- one program reused across chunks -- and included when it's baked (non-rotation).
            kv_pad_rotation_enabled ? std::size_t{0} : logical_n,
            std::cref(ring_size),
            std::cref(compute_kernel_config),
            std::cref(program_config),
            std::cref(ccl_core_grid_offset),
            kv_cache_batch_idx.has_value(),
            kv_pad_rotation_enabled,
            std::cref(latent_v_head_dim),
            std::cref(kv_cache_num_layers),
            std::cref(kv_cache_layer_idx),
            std::cref(all_gather_operation_attributes),
            std::cref(all_gather_tensor_args));
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

    // Trace-safe metadata path (opt-in): uint32 DRAM tensor [slot_id, actual_start, actual_end].
    // metadata[0] = cache-user slot; metadata[1] = kv_actual_isl; metadata[2] is unused by the op.
    // Read via one accessor. std::nullopt => classic host-scalar path.
    std::optional<Tensor> metadata;

    bool has_metadata() const { return metadata.has_value(); }

    // Chunked-prefill is signalled implicitly by Q being shorter than the per-device K shard:
    // Q is the latest slab, K is the populated prefix from chunk 0 through the current chunk.
    uint32_t local_kv_seq_len() const { return static_cast<uint32_t>(input_k.logical_shape()[2]); }

    bool is_chunked() const { return input_q.logical_shape()[2] < local_kv_seq_len(); }

    // Latent-V optimization: absent V means the reader reuses K's buffer
    // and reads the first vDHt head-dim tiles (V's logical head dim).
    bool has_latent_v() const { return !input_v.has_value(); }

    uint32_t v_num_heads() const {
        return input_v.has_value() ? static_cast<uint32_t>(input_v->logical_shape()[1])
                                   : static_cast<uint32_t>(input_k.logical_shape()[1]);
    }

    uint32_t v_head_dim(uint32_t latent_v_head_dim) const {
        return input_v.has_value() ? static_cast<uint32_t>(input_v->logical_shape()[3]) : latent_v_head_dim;
    }
};

// Index constants for RingJointSDPAResult vector
constexpr size_t RING_JOINT_SDPA_OUTPUT_IDX = 0;
constexpr size_t RING_JOINT_SDPA_JOINT_OUTPUT_IDX = 1;
constexpr size_t RING_JOINT_SDPA_STATS_OUTPUT_IDX = 2;

// RingJointSDPAResult is a vector of 3 tensors: [output, joint_output, stats_output]
using RingJointSDPAResult = Tensors;

// RingJointSDPAResultSpec is a vector of 3 TensorSpecs: [output, joint_output, stats_output]
using RingJointSDPAResultSpec = std::vector<TensorSpec>;

}  // namespace ttnn::prim
