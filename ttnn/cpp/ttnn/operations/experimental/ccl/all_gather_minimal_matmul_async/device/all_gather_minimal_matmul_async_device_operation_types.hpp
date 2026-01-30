// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct AllGatherMinimalMatmulAsyncConfig {
    AllGatherMinimalMatmulAsyncConfig(
        uint32_t M_block_size_ = 1,
        uint32_t K_block_size_ = 1,
        uint32_t N_block_size_ = 1,
        uint32_t subblock_h_ = 1,
        uint32_t subblock_w_ = 1,
        CoreCoord compute_with_storage_grid_size_ = {2, 2}) :
        M_block_size(M_block_size_),
        K_block_size(K_block_size_),
        N_block_size(N_block_size_),
        subblock_h(subblock_h_),
        subblock_w(subblock_w_),
        compute_with_storage_grid_size(compute_with_storage_grid_size_) {}

    uint32_t M_block_size;
    uint32_t K_block_size;
    uint32_t N_block_size;
    uint32_t subblock_h;
    uint32_t subblock_w;

    CoreCoord compute_with_storage_grid_size;

    static constexpr auto attribute_names = std::make_tuple(
        "M_block_size", "K_block_size", "N_block_size", "subblock_h", "subblock_w", "compute_with_storage_grid_size");

    auto attribute_values() const {
        return std::forward_as_tuple(
            this->M_block_size,
            this->K_block_size,
            this->N_block_size,
            this->subblock_h,
            this->subblock_w,
            this->compute_with_storage_grid_size);
    }
};

struct AllGatherMinimalMatmulAsyncParams {
    std::optional<const AllGatherMinimalMatmulAsyncConfig> config;
    std::optional<operations::unary::UnaryWithParam> fused_activation;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    uint32_t num_links;
    uint32_t ring_size;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> semaphore;
    std::optional<uint32_t> cluster_axis;
    const std::optional<GlobalSemaphore>& barrier_semaphore;
    bool using_persistent_buffers;
    bool force_transpose;
    uint32_t num_workers_per_link;
    uint32_t num_buffers_per_channel;

    AllGatherMinimalMatmulAsyncParams(
        std::optional<const AllGatherMinimalMatmulAsyncConfig> config,
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
        uint32_t num_buffers_per_channel) :
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
        num_buffers_per_channel(num_buffers_per_channel) {}

    static constexpr auto attribute_names = std::make_tuple(
        "num_links",
        "ring_size",
        "topology",
        "barrier_semaphore",
        "using_persistent_buffers",
        "cluster_axis",
        "semaphore",
        "force_transpose",
        "num_workers_per_link",
        "num_buffers_per_channel");

    auto attribute_values() const {
        return std::forward_as_tuple(
            this->num_links,
            this->ring_size,
            this->topology,
            this->barrier_semaphore,
            this->using_persistent_buffers,
            this->cluster_axis,
            this->semaphore,
            this->force_transpose,
            this->num_workers_per_link,
            this->num_buffers_per_channel);
    }
};

struct AllGatherMinimalMatmulAsyncInputs {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<Tensor> bias_tensor;
    std::optional<Tensor> persistent_output_buffer;
};

}  // namespace ttnn::experimental::prim
