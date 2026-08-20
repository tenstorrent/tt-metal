// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <utility>
#include <vector>
#include <tt_stl/reflection.hpp>

#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::prim {

struct ExpRingJointSDPAParams {
    std::string joint_strategy;
    std::optional<float> scale;
    std::size_t logical_n = 0;
    std::size_t ring_size = 0;
    tt::tt_metal::MemoryConfig output_memory_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // Flattened CCL (all-gather) params
    int32_t dim;
    uint32_t num_links;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    uint32_t cluster_axis;
    uint32_t num_workers_per_link = 1;
    uint32_t num_buffers_per_channel = 8;

    // Sparse computation (frame-block / windowed attention). All three set together or all unset.
    // `tokens_per_frame` is in TOKENS (a multiple of TILE_HEIGHT); `num_frames_padded` is the
    // (sp-aligned) frame count, must be <= 32. `sparse_frame_mask` is a bit-packed row-major
    // [num_frames_padded, num_frames_padded] allow-table: bit (q * num_frames_padded + k) is 1 iff Q
    // frame q attends K frame k (at most 32 uint32 words -> max num_frames_padded = 32). Kept
    // host-side and threaded to the kernels as runtime args (see the program factory). Mirrors the
    // sibling ring_joint_sdpa op.
    std::optional<std::uint32_t> tokens_per_frame = std::nullopt;
    std::optional<std::uint32_t> num_frames_padded = std::nullopt;
    std::vector<std::uint32_t> sparse_frame_mask;  // empty when sparse-frames disabled

    ExpRingJointSDPAParams(
        std::string joint_strategy,
        std::optional<float> scale,
        std::size_t logical_n,
        std::size_t ring_size,
        tt::tt_metal::MemoryConfig output_memory_config,
        std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
        DeviceComputeKernelConfig compute_kernel_config,
        int32_t dim,
        uint32_t num_links,
        ttnn::ccl::Topology topology,
        std::vector<GlobalSemaphore> semaphore,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        uint32_t cluster_axis,
        uint32_t num_workers_per_link = 1,
        uint32_t num_buffers_per_channel = 8,
        std::optional<std::uint32_t> tokens_per_frame = std::nullopt,
        std::optional<std::uint32_t> num_frames_padded = std::nullopt,
        std::vector<std::uint32_t> sparse_frame_mask = {}) :
        joint_strategy(std::move(joint_strategy)),
        scale(scale),
        logical_n(logical_n),
        ring_size(ring_size),
        output_memory_config(std::move(output_memory_config)),
        program_config(std::move(program_config)),
        compute_kernel_config(compute_kernel_config),
        dim(dim),
        num_links(num_links),
        topology(topology),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id),
        cluster_axis(cluster_axis),
        num_workers_per_link(num_workers_per_link),
        num_buffers_per_channel(num_buffers_per_channel),
        tokens_per_frame(tokens_per_frame),
        num_frames_padded(num_frames_padded),
        sparse_frame_mask(std::move(sparse_frame_mask)) {}

    // for Program-cache hash calculation
    static constexpr auto attribute_names = std::forward_as_tuple(
        "joint_strategy",
        "scale",
        "logical_n",
        "ring_size",
        "compute_kernel_config",
        "program_config",
        "dim",
        "num_links",
        "cluster_axis",
        "tokens_per_frame",
        "num_frames_padded",
        "sparse_frame_mask");
    auto attribute_values() const {
        return std::forward_as_tuple(
            joint_strategy,
            scale,
            logical_n,
            ring_size,
            compute_kernel_config,
            program_config,
            dim,
            num_links,
            cluster_axis,
            tokens_per_frame,
            num_frames_padded,
            sparse_frame_mask);
    }

    std::uint32_t get_q_chunk_size() const { return program_config.has_value() ? program_config->q_chunk_size : 32; }

    std::uint32_t get_k_chunk_size() const { return program_config.has_value() ? program_config->k_chunk_size : 32; }

    bool has_sparse_frames() const { return tokens_per_frame.has_value() && num_frames_padded.has_value(); }
};

struct ExpRingJointSDPAInputs {
    Tensor input_q;
    Tensor input_k;
    Tensor input_v;
    // Optional: absent for self-attention. When absent they aren't enumerated as op inputs, avoiding
    // the duplicate-Buffer* footgun that freezes cache-hit addresses (#45452 / #45391). Mirrors ring_joint.
    std::optional<Tensor> joint_q;
    std::optional<Tensor> joint_k;
    std::optional<Tensor> joint_v;
    Tensor gathered_k;
    Tensor gathered_v;
};

// Index constants for ExpRingJointSDPAResult vector
constexpr size_t EXP_RING_JOINT_SDPA_OUTPUT_IDX = 0;
constexpr size_t EXP_RING_JOINT_SDPA_JOINT_OUTPUT_IDX = 1;
constexpr size_t EXP_RING_JOINT_SDPA_STATS_OUTPUT_IDX = 2;

// ExpRingJointSDPAResult is a vector of 3 tensors: [output, joint_output, stats_output]
using ExpRingJointSDPAResult = Tensors;

// ExpRingJointSDPAResultSpec is a vector of 3 TensorSpecs: [output, joint_output, stats_output]
using ExpRingJointSDPAResultSpec = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttnn::prim
