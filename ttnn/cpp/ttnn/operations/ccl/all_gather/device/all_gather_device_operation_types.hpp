// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-umd/types/arch.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::ccl {

// Native algorithm of one all_gather call. Auto appears only in AgForce (no override).
enum class AgAlgo : uint8_t { Auto, Unicast, Multicast, Composite };

// Most unicast workers per link-direction. Measured on BH: n = 4 never beat n = 3 by more than 1%.
inline constexpr uint32_t kAgMaxWorkers = 3;

// Test-only override parsed from TTNN_ALL_GATHER_FORCE (all_gather_plan::parse_force). Zero means "plan's choice".
struct AgForce {
    AgAlgo algo = AgAlgo::Auto;
    uint8_t workers = 0;        // unicast workers per link-direction
    uint8_t links = 0;          // links used
    uint16_t reader_delay = 0;  // cycles a reader spins per slot (sync stress)
    uint16_t writer_delay = 0;  // cycles a writer spins per slot (sync stress)
};

// What the plan decided, computed once per call in all_gather(); the factories re-expand it (all_gather_plan::expand).
struct AgChoice {
    AgAlgo algo = AgAlgo::Composite;
    uint8_t n = 0;      // unicast workers per link-direction; 1 for multicast
    uint8_t links = 0;  // links used
    uint8_t k = 0;      // packets per slot

    bool operator==(const AgChoice&) const = default;
};

// Hardware, fabric and sub-device facts the plan depends on (all_gather_plan::make_facts). Stored in the hashed params
// so a cached program never depends on state it did not hash; injectable in host tests.
struct AgFacts {
    tt::ARCH arch = tt::ARCH::Invalid;
    uint32_t payload = 0;        // fabric max packet payload, bytes
    uint32_t header_bytes = 0;   // fabric packet header, bytes
    uint32_t channel_bytes = 0;  // fabric channel buffer (header + payload), bytes
    uint32_t dram_align = 0;     // DRAM read alignment, bytes
    uint32_t l1_align = 0;       // L1 read alignment, bytes
    uint32_t max_burst = 0;      // NOC_MAX_BURST_SIZE, bytes
    uint32_t num_banks_dram = 0;
    uint32_t num_banks_l1 = 0;
    uint32_t l1_base = 0;           // unreserved L1 base: CBs and the mux map start here
    uint32_t l1_small_floor = 0;    // floor of L1_SMALL: the ceiling of the mux map and of the CB footprint
    uint32_t headers_per_risc = 0;  // fabric packet-header pool entries per data-movement RISC
    std::array<uint32_t, kAgMaxWorkers> mux_map_end{};  // [n - 1]: exclusive end of a Mux V2 map with n channels
    CoreRangeSet worker_cores;                          // sub-device worker cores, intersected with sub_core_grid
};

// The program-cache hash and the profiler's op metadata both come from attribute_names + attribute_values(), every
// struct field must be added to both lists. Miss one and two different configs will share a cached program.
// NOTE: auto-reflection can't be used since it value-initializes the struct, and TensorSpec has no default ctor.
struct AllGatherParams {
    // Gather dim: counted from the end of the shape, so always negative. A padded shape can outrank the
    // logical one (a tiled rank-<2 tensor is promoted to rank 2, extra dims prepended), so this is the only
    // format valid against both. Use Shape::get_normalized_index() where a non-negative axis is needed.
    int32_t dim_from_end = -1;

    tt::tt_metal::TensorSpec output_spec;
    std::optional<uint32_t> cluster_axis;

    // Fabric setup info
    tt::tt_fabric::FabricConfig fabric_config{};
    // Per-axis info (an inactive axis has num_devices = 1, num_links = 0, and Linear topology)
    std::array<tt::tt_fabric::Topology, 2> axis_topology{};
    std::array<bool, 2> axis_is_straight{};  // is the axis wired as a straight physical line
    std::array<uint32_t, 2> axis_num_devices{};
    std::array<uint32_t, 2> axis_num_links{};
    uint32_t num_devices = 0;  // number of devices participating in the collective

    // Worker-core selection.
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
    std::optional<CoreRangeSet> sub_core_grid;

    // Plan inputs and decision (all_gather_plan.hpp). Plain aggregates, so they hash and print field by field through
    // auto-reflection.
    AgFacts facts;
    AgForce force;
    AgChoice choice;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "dim_from_end",
        "output_spec",
        "cluster_axis",
        "fabric_config",
        "axis_topology",
        "axis_is_straight",
        "axis_num_devices",
        "axis_num_links",
        "num_devices",
        "subdevice_id",
        "sub_core_grid",
        "facts",
        "force",
        "choice");
    auto attribute_values() const {
        return std::forward_as_tuple(
            dim_from_end,
            output_spec,
            cluster_axis,
            fabric_config,
            axis_topology,
            axis_is_straight,
            axis_num_devices,
            axis_num_links,
            num_devices,
            subdevice_id,
            sub_core_grid,
            facts,
            force,
            choice);
    }

    // "true 2D" = a 2D fabric config with both axes active
    bool is_true_2d() const {
        return ::tt::tt_fabric::is_2d_fabric_config(fabric_config) && axis_num_devices[0] > 1 &&
               axis_num_devices[1] > 1;
    }

    // The active axis of an effectively-1D topology
    uint32_t get_1d_axis() const {
        TT_FATAL(!is_true_2d(), "get_1d_axis is undefined for true 2D topologies");
        if (cluster_axis.has_value()) {
            return cluster_axis.value();
        }
        for (uint32_t axis = 0; axis < axis_num_devices.size(); ++axis) {
            if (axis_num_devices[axis] > 1) {
                return axis;
            }
        }
        return 0;
    }
};

struct AllGatherInputs {
    Tensor input_tensor;
    std::optional<Tensor> persistent_output_tensor;
};

}  // namespace ttnn::operations::ccl
