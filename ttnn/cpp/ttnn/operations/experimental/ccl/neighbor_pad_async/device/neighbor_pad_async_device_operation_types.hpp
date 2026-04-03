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
#include <tt-metalium/sub_device_types.hpp>

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
    // When true: skip local_copy kernels; output is a compact halo buffer
    // (outer_dim × (padding_left+padding_right) × num_sticks_per_halo_dim sticks).
    // Only the halo rows are written via fabric; the interior is NOT copied.
    // Conv3d reads interior from the original unpadded tensor.
    bool fabric_only = false;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;  // nullopt = use device default

    // Optional progress semaphore: incremented by the H-fabric writer every
    // progress_t_batch_size outer_dim iterations so that conv3d's reader can
    // start processing T-slices before NeighborPad finishes all outer_dims.
    std::optional<GlobalSemaphore> progress_semaphore;
    uint32_t progress_t_batch_size = 0;  // 0 = disabled

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
        std::optional<GlobalSemaphore> progress_semaphore = std::nullopt,
        uint32_t progress_t_batch_size = 0,
        bool fabric_only = false,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt) :
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
        fabric_only(fabric_only),
        sub_device_id(sub_device_id),
        progress_semaphore(progress_semaphore),
        progress_t_batch_size(progress_t_batch_size) {}

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("dim", dim);
        attrs.emplace_back("padding_left", padding_left);
        attrs.emplace_back("padding_right", padding_right);
        attrs.emplace_back("padding_mode", padding_mode);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("h_neighbor_semaphore", h_neighbor_semaphore);
        attrs.emplace_back("w_neighbor_semaphore", w_neighbor_semaphore);
        attrs.emplace_back("barrier_semaphore", barrier_semaphore);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("pad_dim2", pad_dim2);
        attrs.emplace_back("pad2_left", pad2_left);
        attrs.emplace_back("pad2_right", pad2_right);
        attrs.emplace_back("pad2_cluster_axis", pad2_cluster_axis);
        attrs.emplace_back("pad2_num_links", pad2_num_links);
        attrs.emplace_back("using_persistent_buffers", using_persistent_buffers);
        attrs.emplace_back("progress_t_batch_size", progress_t_batch_size);
        attrs.emplace_back("fabric_only", fabric_only);
        return attrs;
    }
};

struct NeighborPadAsyncInputs {
    Tensor input_tensor;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
