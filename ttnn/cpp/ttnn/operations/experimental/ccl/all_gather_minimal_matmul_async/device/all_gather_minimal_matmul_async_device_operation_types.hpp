// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct AllGatherMinimalMatmulAsyncParams {
    std::optional<const MinimalMatmulConfig> config;
    std::optional<operations::unary::UnaryWithParam> fused_activation;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    uint32_t num_links = 0;
    uint32_t ring_size = 0;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> semaphore;
    std::optional<uint32_t> cluster_axis;
    const std::optional<GlobalSemaphore>& barrier_semaphore;
    bool using_persistent_buffers = false;
    bool force_transpose = false;
    uint32_t num_workers_per_link = 0;
    uint32_t num_buffers_per_channel = 0;

    // Fused addcmul: ternary_a + scalar * matmul_output * ternary_b
    std::optional<float> fused_ternary_scalar;

    int32_t chunks = 1;  // Number of output tensors to split into (default 1 for backward compat)
    int32_t dim = -1;    // Dimension to split along (default -1)

    // FSDP fusion: when set, the weight tensor is sharded along its K dim across
    // `fsdp_cluster_axis` with size `fsdp_ring_size`, and the op all-gathers it
    // into `persistent_weight_buffer` before/concurrently-with the matmul.
    std::optional<uint32_t> fsdp_cluster_axis;
    uint32_t fsdp_ring_size = 1;
    std::vector<GlobalSemaphore> fsdp_semaphore;  // ping-pong pair, same shape as `semaphore`
    bool using_persistent_weight_buffer = false;
    ttnn::ccl::Topology fsdp_topology;

    AllGatherMinimalMatmulAsyncParams(
        std::optional<const MinimalMatmulConfig> config,
        std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
        std::optional<tt::tt_metal::MemoryConfig> output_mem_config,
        std::optional<tt::tt_metal::DataType> output_dtype,
        DeviceComputeKernelConfig compute_kernel_config,
        uint32_t num_links,
        uint32_t ring_size,
        ttnn::ccl::Topology topology,
        std::vector<GlobalSemaphore> semaphore,
        std::optional<uint32_t> cluster_axis,
        const std::optional<GlobalSemaphore>& barrier_semaphore,
        bool using_persistent_buffers,
        bool force_transpose,
        uint32_t num_workers_per_link,
        uint32_t num_buffers_per_channel,
        std::optional<float> fused_ternary_scalar,
        int32_t chunks,
        int32_t dim,
        std::optional<uint32_t> fsdp_cluster_axis,
        uint32_t fsdp_ring_size,
        std::vector<GlobalSemaphore> fsdp_semaphore,
        bool using_persistent_weight_buffer,
        ttnn::ccl::Topology fsdp_topology) :
        config(config),
        fused_activation(fused_activation),
        output_mem_config(output_mem_config),
        output_dtype(output_dtype),
        compute_kernel_config(compute_kernel_config),
        num_links(num_links),
        ring_size(ring_size),
        topology(topology),
        semaphore(std::move(semaphore)),
        cluster_axis(cluster_axis),
        barrier_semaphore(barrier_semaphore),
        using_persistent_buffers(using_persistent_buffers),
        force_transpose(force_transpose),
        num_workers_per_link(num_workers_per_link),
        num_buffers_per_channel(num_buffers_per_channel),
        fused_ternary_scalar(fused_ternary_scalar),
        chunks(chunks),
        dim(dim),
        fsdp_cluster_axis(fsdp_cluster_axis),
        fsdp_ring_size(fsdp_ring_size),
        fsdp_semaphore(std::move(fsdp_semaphore)),
        using_persistent_weight_buffer(using_persistent_weight_buffer),
        fsdp_topology(fsdp_topology) {}

    // Structural fields that affect program-cache key.
    //
    // EVERY field the program factory bakes into the kernels must be listed here: two calls whose
    // attributes and tensor specs hash equal reuse the SAME cached program and only re-write its
    // runtime args. chunks/dim/fused_activation/fused_ternary_scalar/output_dtype are all
    // compile-time (chunks is an N_chunks compile arg that sets the output count and the runtime-arg
    // count, fused_activation sets the compute defines, fused_ternary_scalar is baked as a literal,
    // output_dtype picks the CB formats) — omitting them silently aliased ops that differ only in
    // those. That is not hypothetical: with the LTX gate merged into the QKV projection, attn1.to_qkv
    // (4096x4096 local weight, chunks=4, no activation) becomes spec-identical to ffn.ff1
    // (4096x4096, chunks=1, gelu_tanh), so the FFN ran the QKV program — deterministic garbage.
    static constexpr auto attribute_names = std::make_tuple(
        "num_links",
        "ring_size",
        "output_mem_config",
        "output_dtype",
        "topology",
        "cluster_axis",
        "force_transpose",
        "num_workers_per_link",
        "num_buffers_per_channel",
        "config",
        "fused_activation",
        "fused_ternary_scalar",
        "chunks",
        "dim",
        "fsdp_cluster_axis",
        "fsdp_ring_size",
        "using_persistent_weight_buffer");

    auto attribute_values() const {
        return std::forward_as_tuple(
            this->num_links,
            this->ring_size,
            this->output_mem_config,
            this->output_dtype,
            this->topology,
            this->cluster_axis,
            this->force_transpose,
            this->num_workers_per_link,
            this->num_buffers_per_channel,
            this->config,
            this->fused_activation,
            this->fused_ternary_scalar,
            this->chunks,
            this->dim,
            this->fsdp_cluster_axis,
            this->fsdp_ring_size,
            this->using_persistent_weight_buffer);
    }
};

struct AllGatherMinimalMatmulAsyncInputs {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<Tensor> bias_tensor;
    std::optional<Tensor> persistent_output_buffer;

    // Fused addcmul: ternary_a + scalar * matmul_output * ternary_b
    std::optional<Tensor> fused_ternary_input_a;  // residual/base (broadcast like bias)
    std::optional<Tensor> fused_ternary_input_b;  // gate/multiplier (full MxN shape)

    // FSDP fusion: optional pre-allocated buffer holding the gathered weight `[K, N/tp]`.
    // When fsdp_cluster_axis is set on params, this should also be set; the op will
    // populate it with the gathered weight before the matmul reads from it.
    std::optional<Tensor> persistent_weight_buffer;
};

}  // namespace ttnn::experimental::prim
