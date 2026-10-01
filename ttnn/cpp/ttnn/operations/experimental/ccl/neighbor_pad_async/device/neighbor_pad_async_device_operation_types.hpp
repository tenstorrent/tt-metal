// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <optional>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::experimental::prim {

struct NeighborPadAsyncParams {
    uint32_t dim = 0;
    uint32_t padding_left = 0;
    uint32_t padding_right = 0;
    std::string padding_mode;
    uint32_t cluster_axis = 0;
    GlobalSemaphore h_neighbor_semaphore;  // Not default constructible
    GlobalSemaphore w_neighbor_semaphore;  // Not default constructible
    GlobalSemaphore barrier_semaphore;     // Not default constructible
    uint32_t num_links = 0;
    MemoryConfig output_mem_config;
    ttnn::ccl::Topology topology;
    uint32_t ring_size = 0;

    // Secondary dimension for 2D padding (optional)
    std::optional<uint32_t> pad_dim2;
    uint32_t pad2_left = 0;
    uint32_t pad2_right = 0;
    std::optional<uint32_t> pad2_cluster_axis;
    uint32_t pad2_num_links = 0;
    bool using_persistent_buffers = false;
    uint32_t logical_h = 0;  // 0 = no masking; >0 zeros interior rows at global index >= logical_h
    uint32_t t_front_pad = 0;  // 0 = no T-front padding; >0 prepends zero T-frames to output (B=1 only)

    // Constructor required because GlobalSemaphore is not default constructible
    NeighborPadAsyncParams(
        uint32_t dim,
        uint32_t padding_left,
        uint32_t padding_right,
        const std::string& padding_mode,
        uint32_t cluster_axis,
        const GlobalSemaphore& h_neighbor_semaphore,
        const GlobalSemaphore& w_neighbor_semaphore,
        const GlobalSemaphore& barrier_semaphore,
        uint32_t num_links,
        MemoryConfig output_mem_config,
        ttnn::ccl::Topology topology,
        uint32_t ring_size,
        std::optional<uint32_t> pad_dim2 = std::nullopt,
        uint32_t pad2_left = 0,
        uint32_t pad2_right = 0,
        std::optional<uint32_t> pad2_cluster_axis = std::nullopt,
        uint32_t pad2_num_links = 0,
        bool using_persistent_buffers = false,
        uint32_t logical_h = 0,
        uint32_t t_front_pad = 0) :
        dim(dim),
        padding_left(padding_left),
        padding_right(padding_right),
        padding_mode(padding_mode),
        cluster_axis(cluster_axis),
        h_neighbor_semaphore(h_neighbor_semaphore),
        w_neighbor_semaphore(w_neighbor_semaphore),
        barrier_semaphore(barrier_semaphore),
        num_links(num_links),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        ring_size(ring_size),
        pad_dim2(pad_dim2),
        pad2_left(pad2_left),
        pad2_right(pad2_right),
        pad2_cluster_axis(pad2_cluster_axis),
        pad2_num_links(pad2_num_links),
        using_persistent_buffers(using_persistent_buffers),
        logical_h(logical_h),
        t_front_pad(t_front_pad) {}

    // Program-cache hash / canonical-key fields
    static constexpr auto attribute_names = std::make_tuple(
        "dim",
        "padding_left",
        "padding_right",
        "padding_mode",
        "cluster_axis",
        "num_links",
        "output_mem_config",
        "topology",
        "ring_size",
        "pad_dim2",
        "pad2_left",
        "pad2_right",
        "pad2_cluster_axis",
        "pad2_num_links",
        "logical_h",
        "t_front_pad");

    auto attribute_values() const {
        return std::forward_as_tuple(
            this->dim,
            this->padding_left,
            this->padding_right,
            this->padding_mode,
            this->cluster_axis,
            this->num_links,
            this->output_mem_config,
            this->topology,
            this->ring_size,
            this->pad_dim2,
            this->pad2_left,
            this->pad2_right,
            this->pad2_cluster_axis,
            this->pad2_num_links,
            this->logical_h,
            this->t_front_pad);
    }
};

struct NeighborPadAsyncInputs {
    Tensor input_tensor;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
