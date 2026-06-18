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
        uint32_t latent_v_head_dim = 0) :
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
        latent_v_head_dim(latent_v_head_dim) {}

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("joint_strategy", joint_strategy);
        attrs.emplace_back("is_causal", is_causal);
        attrs.emplace_back("is_balanced", is_balanced);
        attrs.emplace_back("is_cross", is_cross);
        attrs.emplace_back("logical_n", logical_n);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_memory_config", output_memory_config);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        attrs.emplace_back("ccl_core_grid_offset", ccl_core_grid_offset);
        if (kv_cache_batch_idx.has_value()) {
            attrs.emplace_back("kv_cache_batch_idx", kv_cache_batch_idx.value());
        }
        if (kv_actual_isl.has_value()) {
            attrs.emplace_back("kv_actual_isl", kv_actual_isl.value());
        }
        attrs.emplace_back("latent_v_head_dim", latent_v_head_dim);
        if (scale.has_value()) {
            attrs.emplace_back("scale", scale);
        }
        if (program_config.has_value()) {
            attrs.emplace_back("program_config", program_config);
        }
        return attrs;
    }

    std::uint32_t get_q_chunk_size() const { return program_config.has_value() ? program_config->q_chunk_size : 32; }

    std::uint32_t get_k_chunk_size() const { return program_config.has_value() ? program_config->k_chunk_size : 32; }

    bool has_indexed_kv_cache() const { return kv_cache_batch_idx.has_value(); }

    bool has_kv_pad_rotation() const { return kv_actual_isl.has_value(); }

    // Program-cache hash / canonical-key fields. Reproduces exactly the field set the former custom
    // compute_program_hash encoded (see ring_joint_sdpa_device_operation.cpp, removed definition):
    //  - cache_key_logical_n: a COMPUTED value (has_kv_pad_rotation() ? 0 : logical_n). When KV-pad
    //    rotation is enabled, logical_n is runtime-patched on a cache hit, so it must NOT be keyed by
    //    its raw value (see validate_runtime_patched_scalars). Returned BY VALUE.
    //  - kv_cache_batch_idx is runtime-patched on a cache hit, so only its PRESENCE (a bool) is keyed,
    //    never the raw value. Returned BY VALUE.
    //  - kv_pad_rotation_enabled == has_kv_pad_rotation() (== kv_actual_isl.has_value()): only the
    //    presence is structural; the raw kv_actual_isl value is runtime-patched. Returned BY VALUE.
    //  - latent_v_head_dim: structural V head dim used when the latent-V optimization is active.
    //  - the nested all-gather contribution: the OLD hash folded in
    //    RingAttentionAllGatherAsyncDeviceOperation::compute_program_hash. We reproduce it by inlining
    //    the all-gather's STRUCTURAL scalar fields {dim, num_links, ring_size, output_mem_config,
    //    topology, sub_device_id, cluster_axis} directly (excludes the raw IDevice* `devices`,
    //    `semaphore`, and `core_allocation_strategy`), plus the all-gather tensor_args on the default
    //    full-tensor walk (a superset, collision-safe). The scalars are inlined (rather than nesting
    //    the whole RingAttentionAllGatherAsyncParams) so this stays self-contained and avoids the
    //    aggregate-vs-compile-time-attributes to_json ambiguity on that struct.
    // The tensor-derived properties the old hash folded in (has_latent_v(), v_num_heads(),
    // v_head_dim()) are NOT reproduced here: they are derived from the RingJointSDPAInputs tensors,
    // which are hashed in full by the default tensor_args walk (a structural superset).
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
        "all_gather_dim",
        "all_gather_num_links",
        "all_gather_ring_size",
        "all_gather_output_mem_config",
        "all_gather_topology",
        "all_gather_sub_device_id",
        "all_gather_cluster_axis",
        "all_gather_tensor_args");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(joint_strategy),
            std::cref(scale),
            std::cref(is_causal),
            std::cref(is_balanced),
            std::cref(is_cross),
            has_kv_pad_rotation() ? std::size_t{0} : logical_n,
            std::cref(ring_size),
            std::cref(compute_kernel_config),
            std::cref(program_config),
            std::cref(ccl_core_grid_offset),
            kv_cache_batch_idx.has_value(),
            has_kv_pad_rotation(),
            std::cref(latent_v_head_dim),
            std::cref(all_gather_operation_attributes.dim),
            std::cref(all_gather_operation_attributes.num_links),
            std::cref(all_gather_operation_attributes.ring_size),
            std::cref(all_gather_operation_attributes.output_mem_config),
            std::cref(all_gather_operation_attributes.topology),
            std::cref(all_gather_operation_attributes.sub_device_id),
            std::cref(all_gather_operation_attributes.cluster_axis),
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
