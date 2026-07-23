// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather {

// The program-cache hash is computed automatically by reflecting over the members
// below and hashing each one. This is safe only because every field here is a
// stable, structural value.
//
// To add a volatile field later (e.g. a semaphore or raw pointer), do not rely
// on this auto-hashing -- define attribute_names + attribute_values() to list
// exactly what to hash.
struct HighBwAllGatherParams {
    int32_t dim = 0;
    MemoryConfig output_mem_config;
    uint32_t cluster_axis = 0;

    // Fabric setup info
    tt::tt_fabric::FabricConfig fabric_config = tt::tt_fabric::FabricConfig::DISABLED;
    // Per-axis info (an inactive axis has num_devices = 1, num_links = 0, and Linear topology)
    std::array<tt::tt_fabric::Topology, 2> axis_topology{};
    std::array<uint32_t, 2> axis_num_devices{};
    std::array<uint32_t, 2> axis_num_links{};
    uint32_t num_devices = 0;  // number of devices participating in the collective
    size_t packet_size = 0;
    // Host-proved structural eligibility for the native store-and-forward
    // transport. Under Fabric2D every logical edge, including ring wrap, must
    // be one direct physical neighbor hop.
    bool neighbor_unicast_eligible = false;
    // Hash of the complete directed physical neighbor plan. Fabric routing
    // arguments are baked into cached programs, so eligibility alone is not a
    // sufficient cache discriminator when the physical plan changes.
    uint64_t neighbor_route_plan_hash = 0;

    // Worker-core selection.
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
    std::optional<CoreRangeSet> sub_core_grid;
};

struct HighBwAllGatherInputs {
    Tensor input_tensor;
    Tensor output_tensor;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather
