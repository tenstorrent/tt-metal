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

struct SolRingParams {
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
    // Mirrors SolRingInputs::chunk_mask having a value. Kept on the params so it reaches the program
    // hash: the mask is a compile-time branch in the kernels, not just data.
    bool has_chunk_mask = false;
    // Experiment knob, not a serving feature: keep every Nth KV chunk and skip the rest, so the skip
    // machinery can be proven and its payoff measured at a controlled density before a real
    // per-dispatch mask exists. 0 = attend every chunk.
    uint32_t chunk_keep_every = 0;

    SolRingParams(
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
        bool has_chunk_mask = false,
        uint32_t chunk_keep_every = 0) :
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
        has_chunk_mask(has_chunk_mask),
        chunk_keep_every(chunk_keep_every) {}

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
        "has_chunk_mask",
        "chunk_keep_every");
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
            has_chunk_mask,
            chunk_keep_every);
    }

    std::uint32_t get_q_chunk_size() const { return program_config.has_value() ? program_config->q_chunk_size : 32; }

    std::uint32_t get_k_chunk_size() const { return program_config.has_value() ? program_config->k_chunk_size : 32; }
};

struct SolRingInputs {
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
    // Optional chunk-skip mask, [H, n_q_chunks, n_kv_chunks_per_ring_iter] packed into uint32 words,
    // ROW_MAJOR. A zero bit means the (q chunk, kv chunk) pair has no selected block and the chunk is
    // skipped outright -- the whole point, since a masked-but-visited chunk costs what an unmasked one
    // does. Selection at chunk granularity only: a block-granular mask needs intra-chunk stamping,
    // which is a later step.
    std::optional<Tensor> chunk_mask;
};

// Index constants for SolRingResult vector
constexpr size_t SOL_RING_OUTPUT_IDX = 0;
constexpr size_t SOL_RING_JOINT_OUTPUT_IDX = 1;
constexpr size_t SOL_RING_STATS_OUTPUT_IDX = 2;

// SolRingResult is a vector of 3 tensors: [output, joint_output, stats_output]
using SolRingResult = Tensors;

// SolRingResultSpec is a vector of 3 TensorSpecs: [output, joint_output, stats_output]
using SolRingResultSpec = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttnn::prim
