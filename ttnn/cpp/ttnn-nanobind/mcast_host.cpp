// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mcast_host.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <functional>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"

namespace ttnn::mcast_host {

namespace kh = ttnn::kernel_lib::host;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::NOC;
using tt::tt_metal::distributed::MeshDevice;

void py_module_types(nb::module_& mod) {
    nb::enum_<dataflow_kernel_lib::DataReadySignal>(mod, "McastDataReady")
        .value("Flag", dataflow_kernel_lib::DataReadySignal::Flag)
        .value("Counter", dataflow_kernel_lib::DataReadySignal::Counter);
    nb::enum_<dataflow_kernel_lib::TransferMode>(mod, "TransferMode")
        .value("Multicast", dataflow_kernel_lib::TransferMode::Multicast)
        .value("ChainUnicast", dataflow_kernel_lib::TransferMode::ChainUnicast);
    nb::enum_<kh::McastCoreOrder>(mod, "McastCoreOrder")
        .value("RowMajor", kh::McastCoreOrder::RowMajor)
        .value("ColumnMajor", kh::McastCoreOrder::ColumnMajor);
    nb::enum_<kh::McastSenderPlacement>(mod, "McastSenderPlacement")
        .value("Uniform", kh::McastSenderPlacement::Uniform)
        .value("Staggered", kh::McastSenderPlacement::Staggered);

    nb::class_<kh::McastConfig>(mod, "McastConfig");
    nb::class_<kh::McastFixedSenderConfig>(mod, "McastFixedSenderConfig");
    nb::class_<kh::McastRotatingSenderConfig>(mod, "McastRotatingSenderConfig");
    nb::class_<kh::McastSenderGridConfig>(mod, "McastSenderGridConfig");
    nb::class_<kh::McastExplicitSenderConfig>(mod, "McastExplicitSenderConfig");
    nb::class_<kh::Mcast>(mod, "Mcast");
}

void py_module(nb::module_& mod) {
    auto mcast = static_cast<nb::class_<kh::Mcast>>(mod.attr("Mcast"));
    mcast.def(
        "attach",
        [](const kh::Mcast& mcast,
           tt::tt_metal::ProgramDescriptor& descriptor,
           const std::string& prefix,
           const nb::list& kernels) {
            std::vector<std::reference_wrapper<tt::tt_metal::KernelDescriptor>> targets;
            targets.reserve(kernels.size());
            for (auto kernel : kernels) {
                targets.emplace_back(nb::cast<tt::tt_metal::KernelDescriptor&>(kernel));
            }
            mcast.attach(descriptor, prefix, targets);
        },
        nb::arg("descriptor"),
        nb::arg("prefix"),
        nb::arg("kernels"));

    mod.def(
        "attach_absent",
        [](tt::tt_metal::KernelDescriptor& kernel, const std::string& prefix) { kh::attach_absent(kernel, prefix); },
        nb::arg("kernel"),
        nb::arg("prefix"));

    static_cast<nb::class_<kh::McastConfig>>(mod.attr("McastConfig"))
        .def(
            "__init__",
            [](kh::McastConfig* self,
               NOC noc,
               bool handshake,
               std::optional<CoreRangeSet> handshake_cores,
               dataflow_kernel_lib::DataReadySignal data_ready,
               std::optional<uint32_t> base_sem_id,
               std::optional<std::vector<uint32_t>> sem_ids,
               dataflow_kernel_lib::TransferMode irregular_receiver_set_mode) {
                new (self) kh::McastConfig{
                    .noc = noc,
                    .handshake = handshake,
                    .handshake_cores = std::move(handshake_cores),
                    .data_ready = data_ready,
                    .base_sem_id = base_sem_id,
                    .sem_ids = std::move(sem_ids),
                    .irregular_receiver_set_mode = irregular_receiver_set_mode};
            },
            nb::kw_only(),
            nb::arg("noc") = NOC::NOC_0,
            nb::arg("handshake") = true,
            nb::arg("handshake_cores") = std::optional<CoreRangeSet>{},
            nb::arg("data_ready") = dataflow_kernel_lib::DataReadySignal::Flag,
            nb::arg("base_sem_id") = std::optional<uint32_t>{},
            nb::arg("sem_ids") = std::optional<std::vector<uint32_t>>{},
            nb::arg("irregular_receiver_set_mode") = dataflow_kernel_lib::TransferMode::Multicast)
        .def_rw("noc", &kh::McastConfig::noc)
        .def_rw("handshake", &kh::McastConfig::handshake)
        .def_rw("handshake_cores", &kh::McastConfig::handshake_cores)
        .def_rw("data_ready", &kh::McastConfig::data_ready)
        .def_rw("base_sem_id", &kh::McastConfig::base_sem_id)
        .def_rw("sem_ids", &kh::McastConfig::sem_ids)
        .def_rw("irregular_receiver_set_mode", &kh::McastConfig::irregular_receiver_set_mode);

    static_cast<nb::class_<kh::McastFixedSenderConfig>>(mod.attr("McastFixedSenderConfig"))
        .def(
            "__init__",
            [](kh::McastFixedSenderConfig* self, uint32_t sender_index, kh::McastSenderPlacement placement) {
                new (self) kh::McastFixedSenderConfig{sender_index, placement};
            },
            nb::kw_only(),
            nb::arg("sender_index") = 0,
            nb::arg("placement") = kh::McastSenderPlacement::Uniform)
        .def_rw("sender_index", &kh::McastFixedSenderConfig::sender_index)
        .def_rw("placement", &kh::McastFixedSenderConfig::placement);

    static_cast<nb::class_<kh::McastRotatingSenderConfig>>(mod.attr("McastRotatingSenderConfig")).def(nb::init<>());

    static_cast<nb::class_<kh::McastSenderGridConfig>>(mod.attr("McastSenderGridConfig"))
        .def(
            "__init__",
            [](kh::McastSenderGridConfig* self,
               CoreRangeSet sender_cores,
               std::optional<kh::McastCoreOrder> sender_order) {
                new (self) kh::McastSenderGridConfig{std::move(sender_cores), sender_order};
            },
            nb::arg("sender_cores"),
            nb::kw_only(),
            nb::arg("sender_order") = std::optional<kh::McastCoreOrder>{})
        .def_rw("sender_cores", &kh::McastSenderGridConfig::sender_cores)
        .def_rw("sender_order", &kh::McastSenderGridConfig::sender_order);

    static_cast<nb::class_<kh::McastExplicitSenderConfig>>(mod.attr("McastExplicitSenderConfig"))
        .def(
            "__init__",
            [](kh::McastExplicitSenderConfig* self, std::vector<std::vector<CoreCoord>> senders_per_group) {
                new (self) kh::McastExplicitSenderConfig{std::move(senders_per_group)};
            },
            nb::arg("senders_per_group"))
        .def_rw("senders_per_group", &kh::McastExplicitSenderConfig::senders_per_group);

    mcast
        .def(
            "__init__",
            [](kh::Mcast* self,
               MeshDevice* device,
               const kh::McastConfig& config,
               const CoreRangeSet& receivers,
               uint32_t receiver_group_size,
               const kh::McastSenderConfig& sender_config,
               kh::McastCoreOrder receiver_order) {
                new (self) kh::Mcast(*device, config, receivers, receiver_group_size, sender_config, receiver_order);
            },
            nb::arg("device"),
            nb::arg("config"),
            nb::arg("receivers"),
            nb::arg("receiver_group_size"),
            nb::arg("sender_config") = kh::McastFixedSenderConfig{},
            nb::arg("receiver_order") = kh::McastCoreOrder::RowMajor,
            nb::keep_alive<1, 2>())
        .def("participating_cores", &kh::Mcast::participating_cores)
        .def("sender_only_cores", &kh::Mcast::sender_only_cores);
}

}  // namespace ttnn::mcast_host
