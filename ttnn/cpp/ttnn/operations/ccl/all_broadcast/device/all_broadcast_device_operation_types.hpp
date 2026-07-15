// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
namespace ttnn::prim {

struct AllBroadcastParams {
    uint32_t num_links = 0;
    uint32_t ring_size = 0;
    MemoryConfig output_mem_config;
    std::optional<uint32_t> cluster_axis;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    tt::tt_fabric::Topology topology{};
    bool use_l1_small_for_semaphores = false;
    // Row-major composite all-gather can select a single batch and/or a valid
    // height prefix in the broadcast reader.  The outputs are compact; no
    // intermediate unshard or slice tensor is materialized.
    std::optional<uint32_t> batch_slice_idx;
    std::optional<uint32_t> valid_gather_extent;
};

}  // namespace ttnn::prim
