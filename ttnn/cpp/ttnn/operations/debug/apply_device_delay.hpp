// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <optional>
#include <cstdint>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/sub_device.hpp>
#include <ttnn/distributed/types.hpp>

namespace ttnn::operations::debug {

// Applies per-device delays by launching a single-core kernel on each device in the mesh that spins
// for the specified number of cycles. The shape of `delays` must match the mesh view shape
// (rows x cols), e.g. for an 8x4 mesh, delays.size()==8 and delays[r].size()==4.
// If `subdevice_id` is provided, the kernel will be scheduled on a worker core belonging to that subdevice.
void apply_device_delay(
    ttnn::MeshDevice& mesh_device,
    const std::vector<std::vector<uint32_t>>& delays,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt);

}  // namespace ttnn::operations::debug
