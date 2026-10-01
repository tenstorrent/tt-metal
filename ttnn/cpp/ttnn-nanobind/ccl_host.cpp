// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for the host-side CCL helpers in ttnn/cpp/ttnn/kernel_lib/ccl.

#include "ccl_host.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow_host.hpp"

namespace ttnn::ccl_host {

void py_module_types(nb::module_& mod) {
    nb::class_<ttnn::ccl::dataflow::PacketDims>(mod, "CclPacketDims")
        .def_ro("packet_size_bytes", &ttnn::ccl::dataflow::PacketDims::packet_size_bytes)
        .def_ro("pages_per_packet", &ttnn::ccl::dataflow::PacketDims::pages_per_packet)
        .def_ro("page_segments", &ttnn::ccl::dataflow::PacketDims::page_segments)
        .def_ro("total_packets", &ttnn::ccl::dataflow::PacketDims::total_packets);
    nb::class_<ttnn::ccl::dataflow::DmRoute>(mod, "CclDmRoute")
        .def_ro("num_hops", &ttnn::ccl::dataflow::DmRoute::num_hops)
        .def_ro("is_forward", &ttnn::ccl::dataflow::DmRoute::is_forward)
        .def_ro("neighbor_id", &ttnn::ccl::dataflow::DmRoute::neighbor_id);
}

void py_module(nb::module_& mod) {
    mod.def(
        "ccl_packet_dims",
        &ttnn::ccl::dataflow::ccl_packet_dims,
        nb::arg("dtype"),
        nb::arg("page_size_bytes"),
        nb::arg("num_pages"),
        nb::arg("alignment"),
        "Frame num_pages of page_size_bytes into fabric packets (owns the bf16 bit_floor + both packing regimes).");
    mod.def(
        "ccl_dm_route",
        [](tt::tt_metal::distributed::MeshDevice* mesh_device,
           const tt::tt_metal::distributed::MeshCoordinate& sender_coord,
           const tt::tt_metal::distributed::MeshCoordinate& receiver_coord,
           ttnn::ccl::Topology topology) {
            return ttnn::ccl::dataflow::ccl_dm_route(mesh_device, sender_coord, receiver_coord, topology);
        },
        nb::arg("mesh_device"),
        nb::arg("sender_coord"),
        nb::arg("receiver_coord"),
        nb::arg("topology"),
        "Compute {num_hops, is_forward, neighbor_id} for a 1-D unicast (owns the fabric fwd/bwd sign reversal + ring "
        "shorter-way, wrap-branch fixed).");
    mod.def(
        "make_ccl_semaphore",
        [](tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t initial_value) {
            return ttnn::ccl::dataflow::make_ccl_semaphore(mesh_device, initial_value);
        },
        nb::arg("mesh_device"),
        nb::arg("initial_value") = 0,
        "Allocate the op-internal cross-device GlobalSemaphore on the mesh's worker cores and run the cache-miss "
        "cross-device Synchronize barrier. Keep the returned handle alive for the cached workload's lifetime (park "
        "it on MeshProgramDescriptor.semaphores).");
    mod.def(
        "build_ccl_fabric_rt_args",
        [](const tt::tt_fabric::FabricNodeId& src_fabric_node_id,
           const tt::tt_fabric::FabricNodeId& neighbor_fabric_node_id,
           uint32_t link_idx,
           tt::tt_metal::ProgramDescriptor& descriptor,
           const tt::tt_metal::CoreCoord& worker_core,
           bool is_forward) {
            return ttnn::ccl::dataflow::build_ccl_fabric_rt_args(
                src_fabric_node_id, neighbor_fabric_node_id, link_idx, descriptor, worker_core, is_forward);
        },
        nb::arg("src_fabric_node_id"),
        nb::arg("neighbor_fabric_node_id"),
        nb::arg("link_idx"),
        nb::arg("program_descriptor"),
        nb::arg("worker_core"),
        nb::arg("is_forward"),
        "Build the complete fabric-connection runtime-arg block in the layout FabricStreamSender consumes "
        "([has_forward][fwd conn][has_backward][bwd conn]) — one call instead of hand-rolling the flag dance "
        "around setup_fabric_connection. Place the block FIRST in the kernel's runtime args. Mutates "
        "program_descriptor (appends SemaphoreDescriptors), same as setup_fabric_connection.");
}

}  // namespace ttnn::ccl_host
