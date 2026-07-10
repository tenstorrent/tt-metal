// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/sub_device_types.hpp>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

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

    // Compile-time attributes drive the default program-cache hash and canonical key.
    // `semaphore` (GlobalSemaphore) is excluded: it is not hashable/canonical-key serializable and
    // is used only for runtime args (semaphore.address()), so it does not affect program structure.
    static constexpr auto attribute_names = std::forward_as_tuple(
        "num_links",
        "ring_size",
        "all_reduce_mem_config",
        "topology",
        "sub_device_id",
        "head_dim",
        "use_noc1_only",
        "num_heads",
        "num_kv_heads",
        "input_on_subcoregrids",
        "slice_size",
        "final_mem_config",
        "dtype",
        "cluster_axis");
    auto attribute_values() const {
        return std::forward_as_tuple(
            num_links,
            ring_size,
            all_reduce_mem_config,
            topology,
            sub_device_id,
            head_dim,
            use_noc1_only,
            num_heads,
            num_kv_heads,
            input_on_subcoregrids,
            slice_size,
            final_mem_config,
            dtype,
            cluster_axis);
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
