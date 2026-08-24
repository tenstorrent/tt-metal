// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Python binding for the mcast HOST helper (ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp).
// Thin passthrough: the API lives entirely in C++; this file only binds the family/group model and
// its Mcast1D/Mcast2D convenience wrappers. There are no Python-side factories or argument rewrites.

#include "mcast_host.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp"

namespace ttnn::mcast_host {

namespace kh = ttnn::kernel_lib::host;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::NOC;
using tt::tt_metal::distributed::MeshDevice;

void py_module_types(nb::module_& mod) {
    nb::enum_<kh::DataReadyMode>(mod, "McastDataReady")
        .value("Flag", kh::DataReadyMode::Flag)
        .value("Counter", kh::DataReadyMode::Counter);
    nb::enum_<kh::Mcast1DShape>(mod, "Mcast1DShape")
        .value("PerRow", kh::Mcast1DShape::PerRow)
        .value("PerColumn", kh::Mcast1DShape::PerColumn);
    nb::enum_<kh::Mcast1DSenderPlacement>(mod, "Mcast1DSenderPlacement")
        .value("Uniform", kh::Mcast1DSenderPlacement::Uniform)
        .value("Diagonal", kh::Mcast1DSenderPlacement::Diagonal);
    nb::class_<kh::McastConfig>(mod, "McastConfig");
    nb::class_<kh::McastGroup>(mod, "McastGroup");
    nb::class_<kh::McastFamily>(mod, "McastFamily");
    nb::class_<kh::Mcast1D>(mod, "Mcast1D");
    nb::class_<kh::Mcast2D>(mod, "Mcast2D");
}

void py_module(nb::module_& mod) {
    // McastConfig — keyword-constructible; every field optional with the C++ default.
    static_cast<nb::class_<kh::McastConfig>>(mod.attr("McastConfig"))
        .def(
            "__init__",
            [](kh::McastConfig* self,
               NOC noc,
               bool handshake,
               kh::DataReadyMode data_ready,
               bool rotating_sender,
               uint32_t base_sem_id,
               std::optional<std::vector<uint32_t>> sem_ids) {
                new (self)
                    kh::McastConfig{noc, handshake, data_ready, rotating_sender, base_sem_id, std::move(sem_ids)};
            },
            nb::kw_only(),
            nb::arg("noc") = NOC::NOC_0,
            nb::arg("handshake") = true,
            nb::arg("data_ready") = kh::DataReadyMode::Flag,
            nb::arg("rotating_sender") = false,
            nb::arg("base_sem_id") = 0,
            nb::arg("sem_ids") = std::optional<std::vector<uint32_t>>{})
        .def_rw("noc", &kh::McastConfig::noc)
        .def_rw("handshake", &kh::McastConfig::handshake)
        .def_rw("data_ready", &kh::McastConfig::data_ready)
        .def_rw("rotating_sender", &kh::McastConfig::rotating_sender)
        .def_rw("base_sem_id", &kh::McastConfig::base_sem_id)
        .def_rw("sem_ids", &kh::McastConfig::sem_ids);

    static_cast<nb::class_<kh::McastGroup>>(mod.attr("McastGroup"))
        .def(
            "__init__",
            [](kh::McastGroup* self,
               const CoreRangeSet& receiver_set,
               const CoreCoord& sender,
               bool use_chain_forwarding,
               std::optional<uint32_t> ack_count) {
                new (self) kh::McastGroup(receiver_set, sender, use_chain_forwarding, ack_count);
            },
            nb::arg("receiver_set"),
            nb::arg("sender"),
            nb::kw_only(),
            nb::arg("use_chain_forwarding") = false,
            nb::arg("ack_count") = std::optional<uint32_t>{})
        .def(
            "__init__",
            [](kh::McastGroup* self,
               const CoreRangeSet& receiver_set,
               const std::vector<CoreCoord>& rotating_senders,
               bool use_chain_forwarding,
               std::optional<uint32_t> ack_count) {
                new (self) kh::McastGroup(receiver_set, rotating_senders, use_chain_forwarding, ack_count);
            },
            nb::arg("receiver_set"),
            nb::arg("rotating_senders"),
            nb::kw_only(),
            nb::arg("use_chain_forwarding") = false,
            nb::arg("ack_count") = std::optional<uint32_t>{})
        .def_prop_ro("receiver_set", &kh::McastGroup::receiver_set)
        .def_prop_ro("senders", &kh::McastGroup::senders)
        .def_prop_ro("rotating_sender", &kh::McastGroup::rotating_sender)
        .def_prop_ro("use_chain_forwarding", &kh::McastGroup::use_chain_forwarding)
        .def_prop_ro("ack_count", &kh::McastGroup::ack_count);

    static_cast<nb::class_<kh::McastFamily>>(mod.attr("McastFamily"))
        .def(
            "__init__",
            [](kh::McastFamily* self,
               MeshDevice* device,
               const std::vector<kh::McastGroup>& groups,
               const kh::McastConfig& config) { new (self) kh::McastFamily(device, groups, config); },
            nb::arg("device"),
            nb::arg("groups"),
            nb::arg("config") = kh::McastConfig{})
        .def("owned_semaphores", &kh::McastFamily::owned_semaphores)
        .def("compile_time_args", &kh::McastFamily::compile_time_args, nb::arg("pre_handshake") = std::optional<bool>{})
        .def("runtime_args", &kh::McastFamily::runtime_args, nb::arg("core"))
        .def("is_sender", &kh::McastFamily::is_sender, nb::arg("core"))
        .def("num_receivers", &kh::McastFamily::num_receivers, nb::arg("core"))
        .def("next_base_sem_id", &kh::McastFamily::next_base_sem_id)
        .def_prop_ro("has_receivers", &kh::McastFamily::has_receivers)
        .def_prop_ro("max_rectangles", &kh::McastFamily::max_rectangles)
        .def_prop_ro("uses_compact_wire", &kh::McastFamily::uses_compact_wire)
        .def_prop_ro("receiver_cores", &kh::McastFamily::receiver_cores)
        .def_prop_ro("participating_cores", &kh::McastFamily::participating_cores)
        .def_prop_ro("sender_only_cores", &kh::McastFamily::sender_only_cores);

    // Mcast1D — the one host helper. Ctor takes the shape enum + config directly (no factories). The
    // Python device is a MeshDevice; the C++ ctor takes IDevice* (upcast at the call).
    static_cast<nb::class_<kh::Mcast1D>>(mod.attr("Mcast1D"))
        .def(
            "__init__",
            [](kh::Mcast1D* self,
               MeshDevice* device,
               const CoreRangeSet& grid,
               kh::Mcast1DShape shape,
               uint32_t sender_index,
               const kh::McastConfig& config) { new (self) kh::Mcast1D(device, grid, shape, sender_index, config); },
            nb::arg("device"),
            nb::arg("grid"),
            nb::arg("shape"),
            nb::arg("sender_index") = 0,
            nb::arg("config") = kh::McastConfig{})
        .def(
            "__init__",
            [](kh::Mcast1D* self,
               MeshDevice* device,
               const CoreRangeSet& grid,
               kh::Mcast1DShape shape,
               uint32_t starting_sender_index,
               kh::Mcast1DSenderPlacement sender_placement,
               const kh::McastConfig& config) {
                new (self) kh::Mcast1D(device, grid, shape, starting_sender_index, config, sender_placement);
            },
            nb::arg("device"),
            nb::arg("grid"),
            nb::arg("shape"),
            nb::arg("starting_sender_index"),
            nb::arg("sender_placement"),
            nb::arg("config") = kh::McastConfig{})
        .def(
            "__init__",
            [](kh::Mcast1D* self,
               MeshDevice* device,
               const CoreRangeSet& receiver_grid,
               kh::Mcast1DShape shape,
               const std::vector<std::vector<CoreCoord>>& rotating_senders,
               const kh::McastConfig& config) {
                new (self) kh::Mcast1D(device, receiver_grid, shape, rotating_senders, config);
            },
            nb::arg("device"),
            nb::arg("receiver_grid"),
            nb::arg("shape"),
            nb::arg("rotating_senders"),
            nb::arg("config") = kh::McastConfig{})
        .def(
            "owned_semaphores",
            &kh::Mcast1D::owned_semaphores,
            R"doc(The SemaphoreDescriptors this helper created, for the factory to add (empty if sem_ids were adopted).)doc")
        .def(
            "compile_time_args",
            &kh::Mcast1D::compile_time_args,
            nb::arg("pre_handshake") = std::optional<bool>{},
            R"doc(Uniform mcast config for the reader CT list: [has_receivers, data_ready_sem_id, consumer_ready_sem_id, ack_count, flags, rotating_span]. Six words, matching McastArgs.)doc")
        .def(
            "runtime_args",
            &kh::Mcast1D::runtime_args,
            nb::arg("core"),
            R"doc(Per-core runtime args. Fixed: 6 words. Rotating: 6 + 2*rotating_span words. The final two words encode this core's roles and sender phase.)doc")
        .def("is_sender", &kh::Mcast1D::is_sender, nb::arg("core"))
        .def("num_receivers", &kh::Mcast1D::num_receivers, nb::arg("core"))
        .def(
            "ack_count",
            &kh::Mcast1D::ack_count,
            R"doc(The sender's handshake ACK wait-count on the wire (Mcast1D is always dense: the EXCLUDE fan-out span-1).)doc")
        .def(
            "next_base_sem_id",
            &kh::Mcast1D::next_base_sem_id,
            R"doc(base_sem_id the next family on the same grid should use so their ids don't overlap.)doc")
        .def("has_receivers", &kh::Mcast1D::has_receivers);

    // Mcast2D — one exact receiver group. Dense sets retain the compact one-rectangle wire; irregular
    // sets use the family multi-rectangle wire. ack_count is the handshake override (0 => fan-out).
    static_cast<nb::class_<kh::Mcast2D>>(mod.attr("Mcast2D"))
        .def(
            "__init__",
            [](kh::Mcast2D* self,
               MeshDevice* device,
               const CoreRangeSet& mcast_rect,
               const CoreCoord& sender,
               const kh::McastConfig& config,
               uint32_t ack_count) { new (self) kh::Mcast2D(device, mcast_rect, sender, config, ack_count); },
            nb::arg("device"),
            nb::arg("mcast_rect"),
            nb::arg("sender"),
            nb::arg("config") = kh::McastConfig{},
            nb::arg("ack_count") = 0)
        .def(
            "__init__",
            [](kh::Mcast2D* self,
               MeshDevice* device,
               const CoreRangeSet& mcast_rect,
               const std::vector<CoreCoord>& rotating_senders,
               const kh::McastConfig& config) { new (self) kh::Mcast2D(device, mcast_rect, rotating_senders, config); },
            nb::arg("device"),
            nb::arg("mcast_rect"),
            nb::arg("rotating_senders"),
            nb::arg("config") = kh::McastConfig{})
        .def(
            "owned_semaphores",
            &kh::Mcast2D::owned_semaphores,
            R"doc(The SemaphoreDescriptors this helper created, placed on the participating set (rect, or rect ∪ {sender}); empty if sem_ids were adopted.)doc")
        .def(
            "compile_time_args",
            &kh::Mcast2D::compile_time_args,
            nb::arg("pre_handshake") = std::optional<bool>{},
            R"doc(Uniform mcast config for the reader CT list: [has_receivers, data_ready_sem_id, consumer_ready_sem_id, ack_count, flags, rotating_span]. Six words, matching McastArgs.)doc")
        .def(
            "runtime_args",
            &kh::Mcast2D::runtime_args,
            nb::arg("core"),
            R"doc(Per-core runtime args. Fixed: 6 words. Rotating: 6 + 2*rotating_span words. The final two words encode this core's roles and sender phase.)doc")
        .def("is_sender", &kh::Mcast2D::is_sender, nb::arg("core"))
        .def("num_receivers", &kh::Mcast2D::num_receivers, nb::arg("core"))
        .def(
            "ack_count",
            &kh::Mcast2D::ack_count,
            R"doc(The handshake ack wait-count on the wire (= fan-out when dense, smaller when divergent).)doc")
        .def(
            "next_base_sem_id",
            &kh::Mcast2D::next_base_sem_id,
            R"doc(base_sem_id the next family on the same grid should use so their ids don't overlap.)doc")
        .def(
            "sender_in_rect",
            &kh::Mcast2D::sender_in_rect,
            R"doc(True if the sender sits inside the rect (fully-inside mode) vs is a separate core.)doc")
        .def("has_receivers", &kh::Mcast2D::has_receivers);
}

}  // namespace ttnn::mcast_host
