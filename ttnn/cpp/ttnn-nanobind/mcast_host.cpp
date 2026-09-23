// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Python binding for the mcast HOST helper (ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp).
// Thin passthrough to the C++ family and rectangular wrappers. Python families use
// the same constructor/add_group/prepare_arguments lifecycle as C++.

#include "mcast_host.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include "ttnn-nanobind/metal2_casters.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
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

template <typename Family>
void bind_descriptor_attach(nb::class_<Family> cls) {
    cls.def(
        "attach",
        [](const Family& family,
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
        nb::arg("kernels"),
        "Append multicast arguments to these kernel objects and allocate or adopt descriptor semaphores. "
        "Assign the completed objects to descriptor.kernels after attachment.");
}

// Keep attachment in C++ so the native transaction updates the Python-owned value objects.
template <typename Family>
void bind_spec_attach(nb::class_<Family> cls) {
    namespace m2 = tt::tt_metal::experimental;
    cls.def(
        "attach",
        [](const Family& family,
           m2::ProgramSpec& spec,
           m2::ProgramRunArgs& run_args,
           const std::string& prefix,
           const std::vector<m2::KernelSpecName>& kernels,
           const std::vector<m2::SemaphoreSpecName>& adopted_semaphores) {
            family.attach(spec, run_args, prefix, kernels, adopted_semaphores);
        },
        nb::arg("spec"),
        nb::arg("run_args"),
        nb::arg("prefix"),
        nb::arg("kernels"),
        nb::arg("adopted_semaphores") = std::vector<m2::SemaphoreSpecName>{},
        "Attach named multicast resources and arguments to placed ProgramSpec kernels.");
}

void py_module_types(nb::module_& mod) {
    nb::enum_<dataflow_kernel_lib::DataReadySignal>(mod, "McastDataReady")
        .value("Flag", dataflow_kernel_lib::DataReadySignal::Flag)
        .value("Counter", dataflow_kernel_lib::DataReadySignal::Counter);
    nb::enum_<dataflow_kernel_lib::TransferMode>(mod, "TransferMode")
        .value("Multicast", dataflow_kernel_lib::TransferMode::Multicast)
        .value("ChainUnicast", dataflow_kernel_lib::TransferMode::ChainUnicast);
    nb::enum_<kh::Mcast1DShape>(mod, "Mcast1DShape")
        .value("PerRow", kh::Mcast1DShape::PerRow)
        .value("PerColumn", kh::Mcast1DShape::PerColumn);
    nb::enum_<kh::Mcast1DSenderPlacement>(mod, "Mcast1DSenderPlacement")
        .value("Uniform", kh::Mcast1DSenderPlacement::Uniform)
        .value("Diagonal", kh::Mcast1DSenderPlacement::Diagonal);
    nb::enum_<kh::Mcast2DSenderOrder>(mod, "Mcast2DSenderOrder")
        .value("RowMajor", kh::Mcast2DSenderOrder::RowMajor)
        .value("ColumnMajor", kh::Mcast2DSenderOrder::ColumnMajor);
    nb::class_<kh::McastConfig>(mod, "McastConfig");
    nb::class_<kh::Mcast1DFixedSenderConfig>(mod, "Mcast1DFixedSenderConfig");
    nb::class_<kh::Mcast1DRotatingSenderConfig>(mod, "Mcast1DRotatingSenderConfig");
    nb::class_<kh::Mcast2DFixedSenderConfig>(mod, "Mcast2DFixedSenderConfig");
    nb::class_<kh::Mcast2DRotatingSenderConfig>(mod, "Mcast2DRotatingSenderConfig");
    nb::class_<kh::McastFamily>(mod, "McastFamily");
    nb::class_<kh::Mcast1D>(mod, "Mcast1D");
    nb::class_<kh::Mcast2D>(mod, "Mcast2D");
}

void py_module(nb::module_& mod) {
    namespace m2 = tt::tt_metal::experimental;
    bind_spec_attach(static_cast<nb::class_<kh::McastFamily>>(mod.attr("McastFamily")));
    bind_spec_attach(static_cast<nb::class_<kh::Mcast1D>>(mod.attr("Mcast1D")));
    bind_spec_attach(static_cast<nb::class_<kh::Mcast2D>>(mod.attr("Mcast2D")));
    mod.def(
        "attach_absent",
        [](m2::ProgramSpec& spec, const std::string& prefix, const std::vector<m2::KernelSpecName>& kernels) {
            kh::attach_absent(spec, prefix, kernels);
        },
        nb::arg("spec"),
        nb::arg("prefix"),
        nb::arg("kernels"),
        "Attach an absent multicast channel without allocating semaphore or runtime resources.");

    bind_descriptor_attach(static_cast<nb::class_<kh::McastFamily>>(mod.attr("McastFamily")));
    bind_descriptor_attach(static_cast<nb::class_<kh::Mcast1D>>(mod.attr("Mcast1D")));
    bind_descriptor_attach(static_cast<nb::class_<kh::Mcast2D>>(mod.attr("Mcast2D")));
    mod.def(
        "attach_absent",
        [](tt::tt_metal::KernelDescriptor& kernel, const std::string& prefix) { kh::attach_absent(kernel, prefix); },
        nb::arg("kernel"),
        nb::arg("prefix"),
        "Append an absent multicast tag and publish its named offsets; no runtime args or semaphores are added.");

    mod.attr("MCAST_ACK_EQUALS_FANOUT") = kh::ACK_EQUALS_FANOUT;

    // McastConfig — keyword-constructible; every field optional with the C++ default.
    static_cast<nb::class_<kh::McastConfig>>(mod.attr("McastConfig"))
        .def(
            "__init__",
            [](kh::McastConfig* self,
               NOC noc,
               bool handshake,
               dataflow_kernel_lib::DataReadySignal data_ready,
               std::optional<uint32_t> base_sem_id,
               std::optional<std::vector<uint32_t>> sem_ids,
               std::optional<uint32_t> ack_count_override,
               dataflow_kernel_lib::TransferMode irregular_receiver_set_mode) {
                new (self) kh::McastConfig{
                    .noc = noc,
                    .handshake = handshake,
                    .data_ready = data_ready,
                    .base_sem_id = base_sem_id,
                    .sem_ids = std::move(sem_ids),
                    .ack_count_override = ack_count_override,
                    .irregular_receiver_set_mode = irregular_receiver_set_mode};
            },
            nb::kw_only(),
            nb::arg("noc") = NOC::NOC_0,
            nb::arg("handshake") = true,
            nb::arg("data_ready") = dataflow_kernel_lib::DataReadySignal::Flag,
            nb::arg("base_sem_id") = nb::none(),
            nb::arg("sem_ids") = std::optional<std::vector<uint32_t>>{},
            nb::arg("ack_count_override") = std::optional<uint32_t>{},
            nb::arg("irregular_receiver_set_mode") = dataflow_kernel_lib::TransferMode::Multicast)
        .def_rw("noc", &kh::McastConfig::noc)
        .def_rw("handshake", &kh::McastConfig::handshake)
        .def_rw("data_ready", &kh::McastConfig::data_ready)
        .def_prop_rw(
            "base_sem_id",
            [](const kh::McastConfig& cfg) { return cfg.base_sem_id.value_or(0); },
            [](kh::McastConfig& cfg, uint32_t value) { cfg.base_sem_id = value; })
        .def_rw(
            "sem_ids",
            &kh::McastConfig::sem_ids,
            "Adopt data_ready, consumer_ready, and (for ChainUnicast) a distinct signal_source ID. "
            "Initialize chain cells to zero; signal_source must not alias another live channel.")
        .def_rw("ack_count_override", &kh::McastConfig::ack_count_override)
        .def_rw("irregular_receiver_set_mode", &kh::McastConfig::irregular_receiver_set_mode);

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
            nb::arg("sender_placement") = kh::Mcast1DSenderPlacement::Uniform)
        .def_rw("starting_sender_index", &kh::Mcast1DFixedSenderConfig::starting_sender_index)
        .def_rw("sender_placement", &kh::Mcast1DFixedSenderConfig::sender_placement);

    static_cast<nb::class_<kh::Mcast1DRotatingSenderConfig>>(mod.attr("Mcast1DRotatingSenderConfig"))
        .def(
            "__init__",
            [](kh::Mcast1DRotatingSenderConfig* self, std::optional<CoreRangeSet> sender_grid) {
                new (self) kh::Mcast1DRotatingSenderConfig{std::move(sender_grid)};
            },
            nb::kw_only(),
            nb::arg("sender_grid") = std::optional<CoreRangeSet>{})
        .def_rw("sender_grid", &kh::Mcast1DRotatingSenderConfig::sender_grid);

    static_cast<nb::class_<kh::Mcast2DFixedSenderConfig>>(mod.attr("Mcast2DFixedSenderConfig"))
        .def(
            "__init__",
            [](kh::Mcast2DFixedSenderConfig* self, const CoreCoord& sender) {
                new (self) kh::Mcast2DFixedSenderConfig{sender};
            },
            nb::arg("sender"))
        .def_rw("sender", &kh::Mcast2DFixedSenderConfig::sender);

    static_cast<nb::class_<kh::Mcast2DRotatingSenderConfig>>(mod.attr("Mcast2DRotatingSenderConfig"))
        .def(
            "__init__",
            [](kh::Mcast2DRotatingSenderConfig* self,
               std::optional<CoreRangeSet> sender_grid,
               kh::Mcast2DSenderOrder sender_order) {
                new (self) kh::Mcast2DRotatingSenderConfig{std::move(sender_grid), sender_order};
            },
            nb::kw_only(),
            nb::arg("sender_grid") = std::optional<CoreRangeSet>{},
            nb::arg("sender_order") = kh::Mcast2DSenderOrder::RowMajor)
        .def_rw("sender_grid", &kh::Mcast2DRotatingSenderConfig::sender_grid)
        .def_rw("sender_order", &kh::Mcast2DRotatingSenderConfig::sender_order);

    static_cast<nb::class_<kh::McastFamily>>(mod.attr("McastFamily"))
        .def(
            "__init__",
            [](kh::McastFamily* self, MeshDevice* device, const kh::McastConfig& config) {
                new (self) kh::McastFamily(device, config);
            },
            nb::arg("device"),
            nb::arg("config") = kh::McastConfig{},
            nb::keep_alive<1, 2>(),
            "Collect groups, then prepare_arguments. Keep the device open through successful argument preparation.")
        .def(
            "add_group",
            &kh::McastFamily::add_group,
            nb::arg("receivers"),
            nb::arg("senders"),
            nb::arg("ack_count_override") = nb::none())
        .def("prepare_arguments", &kh::McastFamily::prepare_arguments)

        .def("participating_cores", &kh::McastFamily::participating_cores)
        .def("sender_only_cores", &kh::McastFamily::sender_only_cores);

    // Mcast1D groups receivers into independent rows or columns. The Python device is a
    // MeshDevice; the C++ constructor takes IDevice* (upcast at the call).
    static_cast<nb::class_<kh::Mcast1D>>(mod.attr("Mcast1D"))
        .def(
            "__init__",
            [](kh::Mcast1D* self,
               MeshDevice* device,
               const CoreRangeSet& receivers,
               kh::Mcast1DShape shape,
               const kh::Mcast1DSenderConfig& sender_config,
               const kh::McastConfig& config) {
                new (self) kh::Mcast1D(device, receivers, shape, sender_config, config);
            },
            nb::arg("device"),
            nb::arg("receivers"),
            nb::arg("shape"),
            nb::arg("sender_config"),
            nb::arg("config") = kh::McastConfig{});

    // Mcast2D configures one rectangular receiver set with fixed or rotating senders.
    static_cast<nb::class_<kh::Mcast2D>>(mod.attr("Mcast2D"))
        .def(
            "__init__",
            [](kh::Mcast2D* self,
               MeshDevice* device,
               const CoreRangeSet& receivers,
               const kh::Mcast2DSenderConfig& sender_config,
               const kh::McastConfig& config) { new (self) kh::Mcast2D(device, receivers, sender_config, config); },
            nb::arg("device"),
            nb::arg("receivers"),
            nb::arg("sender_config"),
            nb::arg("config") = kh::McastConfig{});
}

}  // namespace ttnn::mcast_host
