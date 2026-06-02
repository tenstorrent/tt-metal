// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-nanobind/cluster.hpp"

#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/cluster_noc_helpers.hpp>

#include "ttnn/cluster.hpp"

namespace ttnn::cluster {

namespace {

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

    // ------------------------------------------------------------------------
    // Minimal NOC write / read access to a device core, intended for X280
    // (L2CPU) integration tests that need to poke control mailboxes in LIM
    // from Python without dragging in the `tt_umd` Python bindings (those
    // can't coexist with ttnn in the same process -- both register the same
    // nanobind C++ types and the second import aborts the interpreter).
    //
    // Coordinates are TRANSLATED NOC coords; on Blackhole TRANSLATED == NOC0
    // for L2CPU and worker cores. The data path goes through the same
    // tt::tt_metal::Cluster::write_core / read_core that the H2DSocket /
    // D2HSocket L2CPU constructors use internally.
    // ------------------------------------------------------------------------

    mod.def(
        "write_to_core",
        [](uint32_t device_id, uint32_t x, uint32_t y, uint64_t addr, nb::bytes data) {
            tt::tt_metal::distributed::noc_write(device_id, x, y, addr, std::string_view(data.c_str(), data.size()));
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
                - For L2CPU (X280) targets on Blackhole, ``(x, y)`` is the
                  TRANSLATED NOC coord (e.g. (8, 3) / (8, 5) / (8, 7) / (8, 9))
                  and ``addr`` is the L3 LIM address.
                - The X280 must be initialised (chip not in reset, L2CPU in
                  reset is fine since LIM is NOC-addressable independently
                  of the X280 core itself).
                - ECC priming caveat: a partial-word write to a never-touched
                  LIM cache line can fault. Prime each target line with a
                  64-byte zero write first if writing < 64 B at uninitialised
                  addresses.
        )doc");

    mod.def(
        "read_from_core",
        [](uint32_t device_id, uint32_t x, uint32_t y, uint64_t addr, uint32_t size) -> nb::bytes {
            auto buf = tt::tt_metal::distributed::noc_read(device_id, x, y, addr, size);
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

    // ------------------------------------------------------------------------
    // UC-path counterparts of write_to_core / read_from_core.
    //
    // write_to_core / read_from_core go through UMD's WC TLB window with
    // Relaxed ordering and (above the DMA threshold) the PCIe DMA fast
    // path; both can re-order or merge writes on the host side.
    //
    // *_immediate / *_reg below take the UMD UC TLB / Strict ordering
    // path (Cluster::write_core_immediate / Cluster::read_reg) -- every
    // byte hits the chip in program order with no host-side combining.
    // Use for control registers and any LIM access that must not be
    // merged into a single bursted line.
    // ------------------------------------------------------------------------

    mod.def(
        "write_to_core_immediate",
        [](uint32_t device_id, uint32_t x, uint32_t y, uint64_t addr, nb::bytes data) {
            tt::tt_metal::distributed::noc_write_immediate(
                device_id, x, y, addr, std::string_view(data.c_str(), data.size()));
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
            return tt::tt_metal::distributed::noc_read_reg_u32(device_id, x, y, addr);
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
