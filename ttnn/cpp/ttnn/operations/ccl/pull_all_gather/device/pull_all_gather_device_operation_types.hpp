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

namespace ttnn::operations::ccl {

// The program-cache hash is reflected over the members below, which is safe
// only because every field here is a stable structural value. To add a
// volatile field later (a semaphore, a raw pointer), define attribute_names +
// attribute_values() instead of relying on the auto-hash.
struct PullAllGatherParams {
    int32_t dim = 0;                 // gather dim; this op requires -2, the row dim
    MemoryConfig output_mem_config;  // L1, height sharded
    std::optional<uint32_t> cluster_axis;

    tt::tt_fabric::FabricConfig fabric_config{};
    // Per axis, as AllGatherParams carries them. An inactive axis has
    // num_devices == 1 and Linear topology.
    std::array<tt::tt_fabric::Topology, 2> axis_topology{};
    std::array<uint32_t, 2> axis_num_devices{};
    uint32_t num_devices = 0;
    size_t max_payload_bytes = 0;  // get_tt_fabric_max_payload_size_bytes()

    std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
    std::optional<CoreRangeSet> sub_core_grid;

    // Producer-side and DFB tunables. There is no transaction-ID count here:
    // the counter bank is reserved L1, not an op allocation.
    uint32_t num_producers = 4;
    uint32_t dfb_depth = 8;

    // Peers reached. Used as the barrier fan-in; it is not M, which is the
    // route count.
    uint32_t peer_count() const { return num_devices - 1; }
};

struct PullAllGatherInputs {
    Tensor input_tensor;
    std::optional<Tensor> persistent_output_tensor;
};

}  // namespace ttnn::operations::ccl
