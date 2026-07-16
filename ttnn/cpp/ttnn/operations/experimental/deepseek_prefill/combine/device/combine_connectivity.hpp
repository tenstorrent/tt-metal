// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>  // FabricNodeId
#include <umd/device/types/core_coordinates.hpp>             // tt::CoreType (coord-grid tag for translation)

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace ttnn::operations::experimental::deepseek_prefill::combine {

// [debug] Per-device NoC-connectivity dump for the combine op, captured HOST-side from the program factory.
// It records the on-device NoC data path -- who writes to whom over which NoC -- for the [rxlog]/[txlog] flow-
// control traces to sit alongside.
//
// The machinery is intentionally ROLE-AGNOSTIC: it knows nothing about untilizers / senders / relays / eth.
// The program factory, which already places and wires these cores, hands it a flat list of CoreDesc -- one per
// core it builds -- and the dump prints one line per descriptor. Adding a new kind of tensix core is then just
// a new `type` letter in the descriptor; the connectivity code does not change.

// One core in the combine NoC data path. `type` is a free-form single-letter tag (e.g. 'U' untilizer,
// 'S' sender, 'R' relay, 'E' eth) that the dump prints verbatim and never interprets. `downstream_*` describe
// the on-device NoC write this core makes to the next core in the path. The `fabric_*` fields describe the far
// end of an eth core's fabric (cable) link and are populated ONLY for eth cores -- left as -1 for everyone else.
struct CoreDesc {
    char type = '?';                  // free-form role tag; the dump does not interpret it
    int32_t id = -1;                  // unique within this device
    tt::tt_metal::CoreCoord coord{};  // core's LOGICAL coord; the dump recomputes virtual/noc0/noc1 from it
    // Coord GRID the logical coord lives on (WORKER vs ETH) -- needed only so the dump can translate the logical
    // coord into the other coord systems. This is about the coord grid, NOT the role (`type` stays free-form).
    tt::CoreType core_type = tt::CoreType::WORKER;
    // PHYSICAL NOC0 coord, supplied directly by the caller and populated ONLY for eth cores (from
    // FabricLinkEthInfo::eth_core_noc0). An eth core's logical coord cannot be translated into the other coord
    // systems the way a tensix core's can, so the fabric hands us its physical NOC0 and the dump derives
    // virtual/noc1 from that instead of from `coord`. Left empty for tensix cores (translated from `coord`).
    std::optional<tt::tt_metal::CoreCoord> noc0_physical{};
    // Same-device cores this one writes to over NoC. A vector because fan-out cores (e.g. a sender/relay driving
    // several downstream cores) have more than one; empty => NoC-terminal on this device.
    std::vector<int32_t> downstream_ids{};
    int32_t downstream_noc = -1;  // NoC index (0 == NOC_0, 1 == NOC_1) used for those writes (-1 = n/a)
    // Eth-only (fabric cable) far end; -1 on non-eth cores:
    int32_t fabric_dst_mesh = -1;  // neighbor mesh id this eth forwards to over fabric
    int32_t fabric_dst_dev = -1;   // neighbor chip id this eth forwards to over fabric
    int32_t routing_plane = -1;    // routing plane of this eth's fabric link
};

// Called by the program factory at build time, once per device, with the descriptors it built for that device.
// Stores them in a debug registry keyed by fabric node; the latest build for a node wins. Side-effect-free
// beyond populating the registry.
void record_combine_connectivity(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_fabric::FabricNodeId& src_node,
    std::vector<CoreDesc> cores);

// Called from Python (via nanobind) after the combine op has been built/run: writes one connectivity file per
// device (device_{id}_connectivity.txt) into out_dir. out_dir empty => "generated/combine_flow_log". Devices
// with no captured record (op never built for them) are skipped.
void dump_combine_connectivity(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, const std::string& out_dir = "");

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine
