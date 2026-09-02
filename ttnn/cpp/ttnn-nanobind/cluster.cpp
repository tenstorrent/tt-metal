// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-nanobind/cluster.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <tt-metalium/tt_metal.hpp>
#include <internal/cluster_noc_helpers.hpp>

#include "ttnn/cluster.hpp"

namespace ttnn::cluster {

namespace {

// nb::bytes owns its buffer for the duration of the call, so a non-owning
// span over it is safe for these synchronous writes.
ttsl::Span<const std::byte> as_bytes(const nb::bytes& data) {
    return {reinterpret_cast<const std::byte*>(data.c_str()), data.size()};
}

void bind_ttnn_cluster(nb::module_& mod) {
    mod.def(
        "get_cluster_type",
        &ttnn::cluster::get_cluster_type,
        R"doc(
            Get the cluster type of the current cluster.

            Returns:
                ttnn.cluster.ClusterType: The type of the current cluster.

            Example:
                >>> import ttnn
                >>> cluster_type = ttnn.cluster.get_cluster_type()
                >>> print(cluster_type)
                ttnn.cluster.ClusterType.N150  # (example output)
                >>>
                >>> # You can also compare cluster types
                >>> if cluster_type == ttnn.cluster.ClusterType.T3K:
                ...     print("Running on T3K cluster")
                >>>
                >>> # Or use in conditional logic
                >>> is_galaxy = cluster_type in [ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG]
        )doc");

    mod.def(
        "serialize_cluster_descriptor",
        &ttnn::cluster::serialize_cluster_descriptor,
        R"doc(
            Serialize cluster descriptor to a file.

            Returns:
                str: Path to the serialized cluster descriptor file.

            Example:
                >>> import ttnn
                >>> descriptor_path = ttnn.cluster.serialize_cluster_descriptor()
                >>> print(f"Cluster descriptor saved to: {descriptor_path}")
        )doc");

    mod.def(
        "translate_core_coord",
        [](int device_id, uint32_t x, uint32_t y, const std::string& from_system, const std::string& to_system) {
            return ttnn::cluster::translate_core_coord(device_id, x, y, from_system, to_system);
        },
        nb::arg("device_id"),
        nb::arg("x"),
        nb::arg("y"),
        nb::arg("from_system"),
        nb::arg("to_system"),
        R"doc(
            Translate a core coordinate between coordinate systems for one chip.

            Systems: "LOGICAL", "NOC0", "NOC1", "TRANSLATED".

            device_id is required and not inferred. The mapping is built from that chip's
            harvesting configuration, so translating with another chip's mapping is
            silently wrong -- and on a multi-chip host the harvesting can differ per chip.
            Pass the id of the chip the coordinate actually came from (for profiler data,
            the CSV's PCIe slot).

            Returns:
                tuple[int, int]: The (x, y) pair in `to_system`.

            Example:
                >>> import ttnn
                >>> ttnn.cluster.translate_core_coord(5, 14, 11, "TRANSLATED", "NOC0")
        )doc");

    mod.def(
        "get_chip_unique_id_from_fabric_node_id",
        &ttnn::cluster::get_chip_unique_id_from_fabric_node_id,
        nb::arg("mesh_id"),
        nb::arg("chip_id"),
        R"doc(
            Resolve a FabricNodeId (mesh_id, chip_id) to the chip's hardware-stable 64-bit ASIC unique id.

            This is the chip's physical, host-global-unique identity (the same value fabric sockets
            route by and the migration worker keys per-chip state on). It is NOT the process-local
            logical device id (ttnn.MeshDevice.get_device_id), which collides across the meshes on a host.

            Args:
                mesh_id (int): The fabric mesh id of the node.
                chip_id (int): The fabric chip id of the node within the mesh.

            Returns:
                int: The chip's 64-bit ASIC unique id.

            Example:
                >>> import ttnn
                >>> fnid = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
                >>> unique_id = ttnn.cluster.get_chip_unique_id_from_fabric_node_id(
                ...     int(fnid.mesh_id), int(fnid.chip_id))
        )doc");

    // Raw NOC read/write access to a device core. Coordinates are TRANSLATED NOC
    // coords. Backed by Cluster::write_core / read_core.

    mod.def(
        "write_to_core",
        [](uint32_t device_id, uint32_t x, uint32_t y, uint64_t addr, const nb::bytes& data) {
            tt::tt_metal::internal::noc_write(device_id, x, y, addr, as_bytes(data));
        },
        nb::arg("device_id"),
        nb::arg("x"),
        nb::arg("y"),
        nb::arg("addr"),
        nb::arg("data"),
        R"doc(
            Write raw bytes to a device core over NOC.

            Args:
                device_id (int): Logical chip id.
                x (int): TRANSLATED NOC x coordinate of the target tile.
                y (int): TRANSLATED NOC y coordinate of the target tile.
                addr (int): Device-side address to write to (64-bit).
                data (bytes): Bytes to write.

            Notes:
                - For L2CPU targets, ``addr`` is a LIM address.
                - A partial-line write to LIM whose ECC has not been initialised
                  can fault; write a full 64-byte line first at such addresses.
        )doc");

    mod.def(
        "read_from_core",
        [](uint32_t device_id, uint32_t x, uint32_t y, uint64_t addr, uint32_t size) -> nb::bytes {
            auto buf = tt::tt_metal::internal::noc_read(device_id, x, y, addr, size);
            return nb::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
        },
        nb::arg("device_id"),
        nb::arg("x"),
        nb::arg("y"),
        nb::arg("addr"),
        nb::arg("size"),
        R"doc(
            Read raw bytes from a device core over NOC.

            Args:
                device_id (int): Logical chip id.
                x (int): TRANSLATED NOC x coordinate of the target tile.
                y (int): TRANSLATED NOC y coordinate of the target tile.
                addr (int): Device-side address to read from (64-bit).
                size (int): Number of bytes to read.

            Returns:
                bytes: ``size`` bytes read from ``addr`` on the target tile.

            Notes: See ``write_to_core``.
        )doc");

    // UC-path counterparts. write_to_core / read_from_core use the WC TLB window
    // with Relaxed ordering and may merge or re-order host-side writes; the calls
    // below use the UC TLB window with Strict ordering.

    mod.def(
        "write_to_core_immediate",
        [](uint32_t device_id, uint32_t x, uint32_t y, uint64_t addr, const nb::bytes& data) {
            tt::tt_metal::internal::noc_write_immediate(device_id, x, y, addr, as_bytes(data));
        },
        nb::arg("device_id"),
        nb::arg("x"),
        nb::arg("y"),
        nb::arg("addr"),
        nb::arg("data"),
        R"doc(
            UC-path write: same args as ``write_to_core`` but goes through
            UMD's UC TLB window with Strict ordering. No host-side write
            combining and no DMA fast path; every byte hits the chip in
            program order. Use for control registers and for LIM access
            that must not be merged into bursted lines.
        )doc");

    mod.def(
        "read_reg",
        [](uint32_t device_id, uint32_t x, uint32_t y, uint64_t addr) -> uint32_t {
            return tt::tt_metal::internal::noc_read_reg_u32(device_id, x, y, addr);
        },
        nb::arg("device_id"),
        nb::arg("x"),
        nb::arg("y"),
        nb::arg("addr"),
        R"doc(
            UC-path register read: returns one u32 from ``addr`` on the
            target tile via UMD's UC TLB window with Strict ordering.
            Companion to ``write_to_core_immediate``.
        )doc");

    // Per-bank DRAM NOC routing table, mirroring what
    // RiscFirmwareInitializer::generate_device_bank_to_noc_tables programs for NOC=0.

    mod.def(
        "get_dram_bank_table",
        [](uint32_t device_id) -> nb::list {
            auto table = tt::tt_metal::internal::get_dram_bank_table(device_id);
            nb::list out;
            for (const auto& e : table) {
                nb::dict d;
                d["bank_id"] = e.bank_id;
                d["noc_x"] = e.noc_x;
                d["noc_y"] = e.noc_y;
                d["base_addr"] = e.base_addr;
                d["bank_size"] = e.bank_size;
                out.append(d);
            }
            return out;
        },
        nb::arg("device_id"),
        R"doc(
            Return BRISC-equivalent ``dram_bank_to_noc_xy[NOC0]`` +
            ``bank_to_dram_offset[]`` for an opened device.

            Each entry is a dict with the keys ``bank_id``, ``noc_x``,
            ``noc_y``, ``base_addr``, ``bank_size``. ``noc_x`` / ``noc_y``
            are TRANSLATED on virtualized-DRAM SKUs (Blackhole) and raw
            NOC0 elsewhere -- i.e. the same value the BRISC kernel uses
            when issuing a NOC transaction with NOC_INDEX=0. ``base_addr``
            is the per-bank offset (matches
            ``Allocator::get_bank_offset(BufferType::DRAM, bank_id)``);
            ``bank_size`` is the DRAM view size.

            Args:
                device_id (int): Logical chip id (matches
                    ``IDevice::id()`` / the value used with
                    ``ttnn.open_mesh_device``).

            Returns:
                list[dict]: ``allocator.get_num_banks(DRAM)`` entries,
                indexed by ``bank_id``.

            Notes:
                - The device must be opened first (e.g. via
                  ``ttnn.open_mesh_device``); otherwise this throws.
        )doc");
}

}  // namespace

void py_cluster_module_types(nb::module_& mod) {
    // Bind ClusterType enum using the public API
    nb::enum_<tt::tt_metal::ClusterType>(mod, "ClusterType", "Enum representing different cluster types")
        .value("INVALID", tt::tt_metal::ClusterType::INVALID, "Invalid cluster type")
        .value("N150", tt::tt_metal::ClusterType::N150, "Production N150")
        .value("N300", tt::tt_metal::ClusterType::N300, "Production N300")
        .value("T3K", tt::tt_metal::ClusterType::T3K, "Production T3K, built with 4 N300s")
        .value("GALAXY", tt::tt_metal::ClusterType::GALAXY, "Production Galaxy, all chips with mmio")
        .value("TG", tt::tt_metal::ClusterType::TG, "Will be deprecated")
        .value("P100", tt::tt_metal::ClusterType::P100, "Blackhole single card, ethernet disabled")
        .value("P150", tt::tt_metal::ClusterType::P150, "Blackhole single card, ethernet enabled")
        .value("P150_X2", tt::tt_metal::ClusterType::P150_X2, "2 Blackhole single card, ethernet connected")
        .value("P150_X4", tt::tt_metal::ClusterType::P150_X4, "4 Blackhole single card, ethernet connected")
        .value("P150_X8", tt::tt_metal::ClusterType::P150_X8, "8 Blackhole single card, ethernet connected")
        .value("SIMULATOR_WORMHOLE_B0", tt::tt_metal::ClusterType::SIMULATOR_WORMHOLE_B0, "Simulator Wormhole B0")
        .value("SIMULATOR_BLACKHOLE", tt::tt_metal::ClusterType::SIMULATOR_BLACKHOLE, "Simulator Blackhole")
        .value("N300_2x2", tt::tt_metal::ClusterType::N300_2x2, "2 N300 cards, ethernet connected to form 2x2")
        .value("P300", tt::tt_metal::ClusterType::P300, "Production P300")
        .value("SIMULATOR_QUASAR", tt::tt_metal::ClusterType::SIMULATOR_QUASAR, "Simulator Quasar")
        .value("BLACKHOLE_GALAXY", tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, "Blackhole Galaxy, all chips with mmio")
        .value("P300_X2", tt::tt_metal::ClusterType::P300_X2, "2 P300 cards")
        .value("CUSTOM", tt::tt_metal::ClusterType::CUSTOM, "Custom cluster");
}

void py_cluster_module(nb::module_& mod) { bind_ttnn_cluster(mod); }

}  // namespace ttnn::cluster
