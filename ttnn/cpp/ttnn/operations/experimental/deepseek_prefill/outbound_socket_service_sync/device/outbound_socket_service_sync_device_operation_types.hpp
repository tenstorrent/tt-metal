// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct OutboundSocketServiceSyncParams {
    // Uniform across the mesh (the sender backing shares one page-config on every
    // device).
    uint32_t page_size = 0;
    uint32_t num_pages = 0;
    uint32_t scratch_cb_index = 0;
    uint32_t metadata_size_bytes = 0;  // 0 disables the metadata path
    bool metadata_only = false;        // true skip tensor copy
    tt::tt_metal::CoreRange worker_cores{tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{0, 0}};

    // Per-mesh-coordinate state, indexed row-major as (row * mesh_num_cols + col).
    // The sender service core (and thus the data-ready counter / metadata L1) may
    // differ per device, so these are looked up per coord inside create_descriptor.
    uint32_t mesh_num_cols = 0;
    std::vector<uint32_t> data_ready_addrs;  // L1 addr on each coord's service core
    std::vector<uint32_t> service_core_x;    // LOGICAL service-core x per coord
    std::vector<uint32_t> service_core_y;    // LOGICAL service-core y per coord
    std::vector<uint32_t> metadata_addrs;    // per-coord service-core metadata L1 (metadata mode)
};

struct OutboundSocketServiceSyncInputs {
    // input: the producing stage's output hidden state (read source). Its address
    //   varies per dispatch -> registered as a BufferBinding.
    // backing: the sender service's persistent backing tensor (write dest). Shares
    //   `input`'s per-shard spec; also the op's (in-place) output.
    // metadata: optional [1,1,1,N] uint32 blob the designated worker forwards to the
    //   sender service core (metadata mode only).
    Tensor input;
    Tensor backing;
    std::optional<Tensor> metadata;
};

}  // namespace ttnn::experimental::prim
