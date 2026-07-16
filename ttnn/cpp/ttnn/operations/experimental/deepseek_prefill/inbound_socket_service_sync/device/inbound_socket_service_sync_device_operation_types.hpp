// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct InboundSocketServiceSyncParams {
    // Uniform across the mesh (the backing tensor shares one address/page-config
    // on every device; the data-ready semaphore is a mesh-wide GlobalSemaphore).
    tt::tt_metal::DeviceAddr data_ready_sem_addr = 0;
    uint32_t page_size = 0;
    uint32_t num_pages = 0;
    uint32_t scratch_cb_index = 0;
    uint32_t metadata_size_bytes = 0;  // 0 disables the metadata path
    tt::tt_metal::DeviceAddr metadata_l1_addr = 0;
    CoreRange worker_cores{CoreCoord{0, 0}, CoreCoord{0, 0}};

    // Per-mesh-coordinate state, indexed row-major as (row * mesh_num_cols + col).
    // The service core (and thus the consumed-counter address) may differ per
    // device, so these are looked up per coord inside create_descriptor.
    uint32_t mesh_num_cols = 0;
    std::vector<uint32_t> consumed_addrs;  // L1 addr on each coord's service core
    std::vector<uint32_t> service_core_x;  // LOGICAL service-core x per coord
    std::vector<uint32_t> service_core_y;  // LOGICAL service-core y per coord

    static constexpr auto attribute_names = std::forward_as_tuple(
        "data_ready_sem_addr",
        "page_size",
        "num_pages",
        "scratch_cb_index",
        "metadata_size_bytes",
        "metadata_l1_addr",
        "worker_cores",
        "mesh_num_cols",
        "consumed_addrs",
        "service_core_x",
        "service_core_y");

    auto attribute_values() const {
        return std::forward_as_tuple(
            data_ready_sem_addr,
            page_size,
            num_pages,
            scratch_cb_index,
            metadata_size_bytes,
            metadata_l1_addr,
            worker_cores,
            mesh_num_cols,
            consumed_addrs,
            service_core_x,
            service_core_y);
    }
};

struct InboundSocketServiceSyncInputs {
    // The service's persistent backing tensor (read by the kernel). Supplies the
    // mesh device + per-shard spec; the output tensors mirror its spec.
    const Tensor& backing;

    explicit InboundSocketServiceSyncInputs(const Tensor& backing_in) : backing(backing_in) {}

    static constexpr auto attribute_names = std::forward_as_tuple("backing_dtype");
    auto attribute_values() const { return std::forward_as_tuple(backing.dtype()); }
};

}  // namespace ttnn::experimental::prim
