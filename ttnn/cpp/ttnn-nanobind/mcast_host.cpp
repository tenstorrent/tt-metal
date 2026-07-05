// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Python binding for the mcast HOST helper (ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp).
// Thin passthrough: the API lives entirely in C++ (Mcast1D + McastConfig + Mcast1DShape); this file
// only binds it. No factory functions or arg-massaging here — Python constructs an Mcast1D with the
// shape enum + a McastConfig, exactly as C++ does.

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
    nb::class_<kh::McastConfig>(mod, "McastConfig");
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
            "owned_semaphores",
            &kh::Mcast1D::owned_semaphores,
            R"doc(The SemaphoreDescriptors this helper created, for the factory to add (empty if sem_ids were adopted).)doc")
        .def(
            "compile_time_args",
            &kh::Mcast1D::compile_time_args,
            nb::arg("pre_handshake") = std::optional<bool>{},
            R"doc(Uniform mcast config for the reader CT list: [active, data_ready_sem_id, consumer_ready_sem_id, num_active, flags] (flags bit0=pre_handshake, bit1=data-ready signal). Pass pre_handshake to override the flags bit for this emission (one semantic mcast, per-kernel handshake).)doc")
        .def(
            "runtime_args",
            &kh::Mcast1D::runtime_args,
            nb::arg("core"),
            R"doc(Per-core runtime args. Fixed: 4 words (sender -> dest rect, receiver -> [sender_x, sender_y, 0, 0]). Rotating: 4 + 2*num_senders() words (full-line rect, then one sender coord pair per round).)doc")
        .def("is_sender", &kh::Mcast1D::is_sender, nb::arg("core"))
        .def("num_receivers", &kh::Mcast1D::num_receivers, nb::arg("core"))
        .def(
            "num_active",
            &kh::Mcast1D::num_active,
            R"doc(The sender's handshake ACK wait-count on the wire (Mcast1D is always dense: the EXCLUDE fan-out span-1).)doc")
        .def(
            "num_senders",
            &kh::Mcast1D::num_senders,
            R"doc(Rounds the sender role rotates through (= sender coord pairs in the rotating RT block); 1 in fixed mode.)doc")
        .def(
            "num_semaphores",
            &kh::Mcast1D::num_semaphores,
            R"doc(Semaphores this family created from base_sem_id: 0 (sem_ids adopted) | 1 (no handshake) | 2.)doc")
        .def(
            "next_base_sem_id",
            &kh::Mcast1D::next_base_sem_id,
            R"doc(base_sem_id the next family on the same grid should use so their ids don't overlap.)doc")
        .def("active", &kh::Mcast1D::active);

    // Mcast2D — ONE mcast over a single rectangle. sender ∈ rect => fully-inside (rotating OK,
    // fan-out area-1); sender ∉ rect => separate sender (fixed only, fan-out area). num_active is the
    // handshake ack wait-count (0 => the dense fan-out).
    static_cast<nb::class_<kh::Mcast2D>>(mod.attr("Mcast2D"))
        .def(
            "__init__",
            [](kh::Mcast2D* self,
               MeshDevice* device,
               const CoreRangeSet& mcast_rect,
               const CoreCoord& sender,
               const kh::McastConfig& config,
               uint32_t num_active) { new (self) kh::Mcast2D(device, mcast_rect, sender, config, num_active); },
            nb::arg("device"),
            nb::arg("mcast_rect"),
            nb::arg("sender"),
            nb::arg("config") = kh::McastConfig{},
            nb::arg("num_active") = 0)
        .def(
            "owned_semaphores",
            &kh::Mcast2D::owned_semaphores,
            R"doc(The SemaphoreDescriptors this helper created, placed on the participating set (rect, or rect ∪ {sender}); empty if sem_ids were adopted.)doc")
        .def(
            "compile_time_args",
            &kh::Mcast2D::compile_time_args,
            nb::arg("pre_handshake") = std::optional<bool>{},
            R"doc(Uniform mcast config for the reader CT list: [active, data_ready_sem_id, consumer_ready_sem_id, num_active, flags] (flags bit0=pre_handshake, bit1=data-ready signal). Pass pre_handshake to override the flags bit for this emission (one semantic mcast, per-kernel handshake).)doc")
        .def(
            "runtime_args",
            &kh::Mcast2D::runtime_args,
            nb::arg("core"),
            R"doc(Per-core runtime args. Fixed: 4 words (sender -> dest rect, receiver -> [sender_x, sender_y, 0, 0]). Rotating: 4 + 2*num_senders() words (full-rect rect, then one sender coord pair per round).)doc")
        .def("is_sender", &kh::Mcast2D::is_sender, nb::arg("core"))
        .def("num_receivers", &kh::Mcast2D::num_receivers, nb::arg("core"))
        .def(
            "num_active",
            &kh::Mcast2D::num_active,
            R"doc(The handshake ack wait-count on the wire (= fan-out when dense, smaller when divergent).)doc")
        .def(
            "num_senders",
            &kh::Mcast2D::num_senders,
            R"doc(Rounds the sender role rotates through (= sender coord pairs in the rotating RT block); 1 in fixed mode.)doc")
        .def(
            "num_semaphores",
            &kh::Mcast2D::num_semaphores,
            R"doc(Semaphores this helper created: 0 (sem_ids adopted) | 1 (no handshake) | 2.)doc")
        .def(
            "next_base_sem_id",
            &kh::Mcast2D::next_base_sem_id,
            R"doc(base_sem_id the next family on the same grid should use so their ids don't overlap.)doc")
        .def(
            "sender_in_rect",
            &kh::Mcast2D::sender_in_rect,
            R"doc(True if the sender sits inside the rect (fully-inside mode) vs is a separate core.)doc")
        .def("active", &kh::Mcast2D::active);
}

}  // namespace ttnn::mcast_host
