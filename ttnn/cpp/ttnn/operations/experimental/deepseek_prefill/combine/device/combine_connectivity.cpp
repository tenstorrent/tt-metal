// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_connectivity.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>  // get_fabric_node_id_from_physical_chip_id
#include <umd/device/types/core_coordinates.hpp>       // tt::CoreType

namespace ttnn::operations::experimental::deepseek_prefill::combine {

namespace {

// What the factory captured for one device: the fabric node it was built for and its core descriptors.
struct DeviceConnectivity {
    uint32_t fabric_mesh = 0;
    uint32_t fabric_dev = 0;
    std::vector<CoreDesc> cores;
};

// Debug registry, keyed by fabric node (mesh_id, chip_id). Keyed this way (not by physical chip id) so the
// factory does not need control-plane access to compute the key; dump_combine_connectivity maps each physical
// device back to its fabric node via the public get_fabric_node_id_from_physical_chip_id(). Latest build wins.
std::map<std::pair<uint32_t, uint32_t>, DeviceConnectivity>& registry() {
    static std::map<std::pair<uint32_t, uint32_t>, DeviceConnectivity> r;
    return r;
}

// Human-readable name for a NoC index (0 == NOC_0, 1 == NOC_1); anything else prints verbatim.
std::string noc_name(int32_t noc) { return noc == 0 ? "NOC_0" : (noc == 1 ? "NOC_1" : fmt::format("NOC_{}", noc)); }

// Recompute a core's virtual / physical-NOC0 / physical-NOC1 coord from its descriptor + device context.
// Two paths, depending on whether the caller supplied a physical NOC0 coord:
//   - Tensix cores (no noc0_physical): translate the LOGICAL coord through the grid `core_type` selects
//     (WORKER vs ETH). On the unharvested Blackhole grid a worker's virtual coord IS its physical NOC0, so
//     noc0 is taken from virtual and noc1 is the physical grid mirror (grid-1-noc0).
//   - Eth cores (noc0_physical set): an eth core's logical coord cannot be translated the same way, so the
//     fabric (FabricLinkEthInfo) hands us the true physical NOC0 directly. noc0 comes from there and virt is
//     derived from it (== noc0 on Blackhole, where virtual and physical NOC0 coincide).
struct CoordSet {
    tt::tt_metal::CoreCoord virt;
    tt::tt_metal::CoreCoord noc0;
    tt::tt_metal::CoreCoord noc1;
};
CoordSet coord_systems(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, const tt::tt_metal::CoreCoord& grid, const CoreDesc& c) {
    tt::tt_metal::CoreCoord virt;
    tt::tt_metal::CoreCoord noc0;
    if (c.noc0_physical.has_value()) {
        noc0 = *c.noc0_physical;  // fabric-supplied physical NOC0 (eth cores)
        virt = noc0;              // derive virtual from physical NOC0
    } else {
        virt = mesh_device.virtual_core_from_logical_core(c.coord, c.core_type);
        noc0 = virt;  // unharvested BH: worker virtual == physical NOC0
    }
    const tt::tt_metal::CoreCoord noc1{grid.x - 1 - noc0.x, grid.y - 1 - noc0.y};
    return {virt, noc0, noc1};
}

}  // namespace

void record_combine_connectivity(
    tt::tt_metal::distributed::MeshDevice* /*mesh_device*/,
    const tt::tt_fabric::FabricNodeId& src_node,
    std::vector<CoreDesc> cores) {
    DeviceConnectivity rec;
    rec.fabric_mesh = static_cast<uint32_t>(*src_node.mesh_id);
    rec.fabric_dev = static_cast<uint32_t>(src_node.chip_id);
    rec.cores = std::move(cores);
    registry()[{rec.fabric_mesh, rec.fabric_dev}] = std::move(rec);
}

void dump_combine_connectivity(const tt::tt_metal::distributed::MeshDevice& mesh_device, const std::string& out_dir) {
    const std::filesystem::path dir =
        out_dir.empty() ? std::filesystem::path("generated/combine_flow_log") : std::filesystem::path(out_dir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    const auto grid = mesh_device.grid_size();  // grid context (same for every device of the mesh)

    uint32_t files_written = 0;
    for (tt::tt_metal::IDevice* device : mesh_device.get_devices()) {
        // Map this physical device back to its fabric node to look up what the factory captured.
        const auto node = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device->id());
        const auto key = std::make_pair(static_cast<uint32_t>(*node.mesh_id), static_cast<uint32_t>(node.chip_id));
        auto it = registry().find(key);
        if (it == registry().end()) {
            continue;  // combine op never built for this device -> nothing captured
        }
        const DeviceConnectivity& rec = it->second;

        const auto path = dir / fmt::format("device_{}_connectivity.txt", device->id());
        std::ofstream f(path);
        f << fmt::format(
            "# combine NoC-connectivity trace\n"
            "# device_id={} fabric_node=(mesh={}, dev={}) grid_size=({},{})\n"
            "# one line per core (key=value, python/regex friendly):\n"
            "#   [<type>] id=<id> logical=(x,y) virtual=(x,y) noc0=(x,y) noc1=(x,y) noc=<NOC_x|none>\n"
            "#     down=[<id>,<id>,...] [fabric=(mesh,dev,plane)]\n"
            "#   type is a free-form role tag (e.g. U untilizer, S sender, R relay, E eth). noc0 = physical,\n"
            "#     noc1 = physical mirror, virtual = translated. Tensix coords are recomputed from the logical\n"
            "#     coord; eth coords come from the fabric's physical NOC0 (eth logical coords can't be translated)\n"
            "#   down = the same-device cores this one writes to over noc (empty => NoC-terminal); a core may\n"
            "#     fan out to several. fabric=... is present only for eth ('E') cores (the fabric-cable far end)\n",
            device->id(),
            rec.fabric_mesh,
            rec.fabric_dev,
            grid.x,
            grid.y);

        for (const auto& c : rec.cores) {
            const auto cs = coord_systems(mesh_device, grid, c);
            const bool has_down = !c.downstream_ids.empty();
            std::string down;  // comma-separated id list inside [] (empty brackets => terminal)
            for (size_t i = 0; i < c.downstream_ids.size(); ++i) {
                down += (i ? "," : "") + std::to_string(c.downstream_ids[i]);
            }
            std::string line = fmt::format(
                "[{}] id={} logical=({},{}) virtual=({},{}) noc0=({},{}) noc1=({},{}) noc={} down=[{}]",
                c.type,
                c.id,
                c.coord.x,
                c.coord.y,
                cs.virt.x,
                cs.virt.y,
                cs.noc0.x,
                cs.noc0.y,
                cs.noc1.x,
                cs.noc1.y,
                has_down ? noc_name(c.downstream_noc) : "none",
                down);
            if (c.fabric_dst_dev >= 0) {
                line += fmt::format(" fabric=({},{},{})", c.fabric_dst_mesh, c.fabric_dst_dev, c.routing_plane);
            }
            f << line << '\n';
        }
        ++files_written;
    }

    fmt::print(stderr, "[combine-log] wrote {} connectivity file(s) to {}\n", files_written, dir.string());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine
