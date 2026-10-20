// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Python binding for the matmul_reduce_scatter copy of the mcast host helper
// (ttnn/cpp/ttnn/kernel_lib/mrs/mcast/host/mcast_host.hpp). ProgramDescriptor attachment only.

#include "mrs_mcast_host.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <functional>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/kernel_lib/mrs/mcast/host/mcast_host.hpp"

namespace ttnn::mrs_mcast_host {

namespace kh = ttnn::kernel_lib::mrs_host;
namespace dkl = mrs_dataflow_kernel_lib;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::NOC;
using tt::tt_metal::distributed::MeshDevice;

void py_module_types(nb::module_& mod) {
    nb::enum_<dkl::DataReadySignal>(mod, "McastDataReady")
        .value("Flag", dkl::DataReadySignal::Flag)
        .value("Counter", dkl::DataReadySignal::Counter);
    nb::enum_<dkl::TransferMode>(mod, "TransferMode")
        .value("Multicast", dkl::TransferMode::Multicast)
        .value("ChainUnicast", dkl::TransferMode::ChainUnicast);
    nb::enum_<kh::Mcast1DShape>(mod, "Mcast1DShape")
        .value("PerRow", kh::Mcast1DShape::PerRow)
        .value("PerColumn", kh::Mcast1DShape::PerColumn);
    nb::enum_<kh::Mcast1DSenderPlacement>(mod, "Mcast1DSenderPlacement")
        .value("Uniform", kh::Mcast1DSenderPlacement::Uniform)
        .value("Diagonal", kh::Mcast1DSenderPlacement::Diagonal);
    nb::class_<kh::McastConfig>(mod, "McastConfig");
    nb::class_<kh::Mcast1DFixedSenderConfig>(mod, "Mcast1DFixedSenderConfig");
    nb::class_<kh::Mcast1D>(mod, "Mcast1D");
}

void py_module(nb::module_& mod) {
    mod.def(
        "attach_absent",
        [](tt::tt_metal::KernelDescriptor& kernel, const std::string& prefix) { kh::attach_absent(kernel, prefix); },
        nb::arg("kernel"),
        nb::arg("prefix"));

    static_cast<nb::class_<kh::McastConfig>>(mod.attr("McastConfig"))
        .def(
            "__init__",
            [](kh::McastConfig* self, NOC noc, bool handshake, dkl::DataReadySignal data_ready) {
                new (self) kh::McastConfig{.noc = noc, .handshake = handshake, .data_ready = data_ready};
            },
            nb::kw_only(),
            nb::arg("noc") = NOC::NOC_0,
            nb::arg("handshake") = true,
            nb::arg("data_ready") = dkl::DataReadySignal::Flag);

    static_cast<nb::class_<kh::Mcast1DFixedSenderConfig>>(mod.attr("Mcast1DFixedSenderConfig"))
        .def(
            "__init__",
            [](kh::Mcast1DFixedSenderConfig* self,
               uint32_t starting_sender_index,
               kh::Mcast1DSenderPlacement sender_placement) {
                new (self) kh::Mcast1DFixedSenderConfig{starting_sender_index, sender_placement};
            },
            nb::kw_only(),
            nb::arg("starting_sender_index") = 0,
            nb::arg("sender_placement") = kh::Mcast1DSenderPlacement::Uniform);

    static_cast<nb::class_<kh::Mcast1D>>(mod.attr("Mcast1D"))
        .def(
            "__init__",
            [](kh::Mcast1D* self,
               MeshDevice* device,
               const CoreRangeSet& receivers,
               kh::Mcast1DShape shape,
               const kh::Mcast1DFixedSenderConfig& sender_config,
               const kh::McastConfig& config) {
                new (self) kh::Mcast1D(device, receivers, shape, kh::Mcast1DSenderConfig{sender_config}, config);
            },
            nb::arg("device"),
            nb::arg("receivers"),
            nb::arg("shape"),
            nb::arg("sender_config"),
            nb::arg("config"))
        .def(
            "attach",
            [](const kh::Mcast1D& family,
               tt::tt_metal::ProgramDescriptor& descriptor,
               const std::string& prefix,
               const nb::list& kernels) {
                std::vector<std::reference_wrapper<tt::tt_metal::KernelDescriptor>> targets;
                targets.reserve(kernels.size());
                for (auto kernel : kernels) {
                    targets.emplace_back(nb::cast<tt::tt_metal::KernelDescriptor&>(kernel));
                }
                family.attach(descriptor, prefix, targets);
            },
            nb::arg("descriptor"),
            nb::arg("prefix"),
            nb::arg("kernels"));
}

}  // namespace ttnn::mrs_mcast_host
