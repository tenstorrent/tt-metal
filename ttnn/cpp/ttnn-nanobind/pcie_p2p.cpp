// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pcie_p2p.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/pcie_p2p.hpp>

namespace ttnn::pcie_p2p {

void bind_pcie_p2p_api(nb::module_& mod) {
    namespace p2p = tt::tt_metal::experimental::pcie_p2p;
    mod.def(
        "use_isolated_mesh",
        &p2p::use_isolated_mesh,
        nb::arg("chips"),
        "Declare the given physical chip ids as a 1xN mesh with no Ethernet between them. Call before any device "
        "opens.");
    mod.def(
        "setup",
        [](tt::tt_metal::distributed::MeshDevice* mesh,
           const std::vector<tt::tt_metal::CoreCoord>& cores,
           uint64_t noc_base,
           uint32_t first_iatu_region,
           const std::string& ordering) {
            p2p::Config cfg;
            cfg.noc_base = noc_base;
            cfg.first_iatu_region = first_iatu_region;
            cfg.ordering = ordering;
            p2p::setup(*mesh, cores, cfg);
        },
        nb::arg("mesh_device"),
        nb::arg("cores"),
        nb::arg("noc_base") = (32ull << 30),
        nb::arg("first_iatu_region") = 8,
        nb::arg("ordering") = "relaxed",
        "Program outbound iATU regions and per-core inbound TLB windows for every device in the mesh.");
    mod.def("is_setup", &p2p::is_setup);
    mod.def("teardown", &p2p::teardown);
    mod.def(
        "remote_l1_noc_addr",
        &p2p::remote_l1_noc_addr,
        nb::arg("from_chip"),
        nb::arg("to_chip"),
        nb::arg("core"),
        nb::arg("l1_addr"),
        "64-bit NoC address from which a core on from_chip reaches l1_addr of the given logical core on to_chip.");
    mod.def("mesh_index", &p2p::mesh_index, nb::arg("chip"));
    mod.def("cores", [] { return p2p::cores(); });
}

}  // namespace ttnn::pcie_p2p
