// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/sub_device_types.hpp>
#include <tt_stl/reflection.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

struct AllReduceCreateQkvHeadsParams {
    uint32_t num_links;
    uint32_t ring_size;
    MemoryConfig all_reduce_mem_config;
    ttnn::ccl::Topology topology;
    GlobalSemaphore semaphore;  // Not default constructible
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    uint32_t head_dim;
    bool use_noc1_only;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    bool input_on_subcoregrids;
    std::optional<uint32_t> slice_size;
    MemoryConfig final_mem_config;
    DataType dtype;
    uint32_t cluster_axis;

    // Constructor required because GlobalSemaphore is not default constructible
    AllReduceCreateQkvHeadsParams(
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig all_reduce_mem_config,
        ttnn::ccl::Topology topology,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        uint32_t head_dim,
        bool use_noc1_only,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        bool input_on_subcoregrids,
        std::optional<uint32_t> slice_size,
        MemoryConfig final_mem_config,
        DataType dtype,
        uint32_t cluster_axis) :
        num_links(num_links),
        ring_size(ring_size),
        all_reduce_mem_config(std::move(all_reduce_mem_config)),
        topology(topology),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id),
        head_dim(head_dim),
        use_noc1_only(use_noc1_only),
        num_heads(num_heads),
        num_kv_heads(num_kv_heads),
        input_on_subcoregrids(input_on_subcoregrids),
        slice_size(slice_size),
        final_mem_config(std::move(final_mem_config)),
        dtype(dtype),
        cluster_axis(cluster_axis) {}

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("all_reduce_mem_config", all_reduce_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);
        attrs.emplace_back("head_dim", head_dim);
        attrs.emplace_back("use_noc1_only", use_noc1_only);
        attrs.emplace_back("num_heads", num_heads);
        attrs.emplace_back("num_kv_heads", num_kv_heads);
        attrs.emplace_back("input_on_subcoregrids", input_on_subcoregrids);
        attrs.emplace_back("slice_size", slice_size);
        attrs.emplace_back("final_mem_config", final_mem_config);
        attrs.emplace_back("dtype", dtype);
        attrs.emplace_back("cluster_axis", cluster_axis);
        return attrs;
    }
};

struct AllReduceCreateQkvHeadsInputs {
    Tensor input_tensor;
    Tensor buffer_tensor;
    Tensor batch_offset_tensor;
};

// Return types using named structs for Q, K, V heads
// all_reduce is included as an internal implementation detail
struct AllReduceCreateQkvHeadsResultSpec {
    TensorSpec all_reduce;  // Internal: needed for circular buffer setup
    TensorSpec q;
    TensorSpec k;
    TensorSpec v;
};

struct AllReduceCreateQkvHeadsResult {
    Tensor all_reduce;  // Internal: needed for circular buffer setup
    Tensor q;
    Tensor k;
    Tensor v;
};

}  // namespace ttnn::experimental::prim
