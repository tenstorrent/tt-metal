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
#include <nanobind/stl/variant.h>
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
    nb::enum_<kh::Mcast2DSenderOrder>(mod, "Mcast2DSenderOrder")
        .value("RowMajor", kh::Mcast2DSenderOrder::RowMajor)
        .value("ColumnMajor", kh::Mcast2DSenderOrder::ColumnMajor);
    nb::class_<kh::McastConfig>(mod, "McastConfig");
    nb::class_<kh::Mcast1DFixedSenderConfig>(mod, "Mcast1DFixedSenderConfig");
    nb::class_<kh::Mcast1DRotatingSenderConfig>(mod, "Mcast1DRotatingSenderConfig");
    nb::class_<kh::Mcast2DFixedSenderConfig>(mod, "Mcast2DFixedSenderConfig");
    nb::class_<kh::Mcast2DRotatingSenderConfig>(mod, "Mcast2DRotatingSenderConfig");
    nb::class_<kh::Mcast1D>(mod, "Mcast1D");
    nb::class_<kh::Mcast2D>(mod, "Mcast2D");
}

void py_module(nb::module_& mod) {
    mod.attr("MCAST_ACK_EQUALS_FANOUT") = kh::ACK_EQUALS_FANOUT;

    // McastConfig — keyword-constructible; every field optional with the C++ default.
    static_cast<nb::class_<kh::McastConfig>>(mod.attr("McastConfig"))
        .def(
            "__init__",
            [](kh::McastConfig* self,
               NOC noc,
               bool handshake,
               kh::DataReadyMode data_ready,
               uint32_t base_sem_id,
               std::optional<std::vector<uint32_t>> sem_ids,
               std::optional<uint32_t> ack_count_override) {
                new (self)
                    kh::McastConfig{noc, handshake, data_ready, base_sem_id, std::move(sem_ids), ack_count_override};
            },
            nb::kw_only(),
            nb::arg("noc") = NOC::NOC_0,
            nb::arg("handshake") = true,
            nb::arg("data_ready") = kh::DataReadyMode::Flag,
            nb::arg("base_sem_id") = 0,
            nb::arg("sem_ids") = std::optional<std::vector<uint32_t>>{},
            nb::arg("ack_count_override") = std::optional<uint32_t>{})
        .def_rw("noc", &kh::McastConfig::noc)
        .def_rw("handshake", &kh::McastConfig::handshake)
        .def_rw("data_ready", &kh::McastConfig::data_ready)
        .def_rw("base_sem_id", &kh::McastConfig::base_sem_id)
        .def_rw("sem_ids", &kh::McastConfig::sem_ids)
        .def_rw("ack_count_override", &kh::McastConfig::ack_count_override);

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

    // Mcast1D — independent row or column multicasts with an explicit fixed or rotating sender config.
    static_cast<nb::class_<kh::Mcast1D>>(mod.attr("Mcast1D"))
        .def(
            "__init__",
            [](kh::Mcast1D* self,
               MeshDevice* device,
               const CoreRangeSet& grid,
               kh::Mcast1DShape shape,
               const kh::Mcast1DSenderConfig& sender_config,
               const kh::McastConfig& config) { new (self) kh::Mcast1D(device, grid, shape, sender_config, config); },
            nb::arg("device"),
            nb::arg("grid"),
            nb::arg("shape"),
            nb::arg("sender_config"),
            nb::arg("config") = kh::McastConfig{})
        .def(
            "owned_semaphores",
            &kh::Mcast1D::owned_semaphores,
            R"doc(The SemaphoreDescriptors this helper created, for the factory to add (empty if sem_ids were adopted).)doc")
        .def(
            "compile_time_args",
            &kh::Mcast1D::compile_time_args,
            nb::arg("pre_handshake") = std::optional<bool>{},
            R"doc(McastArgs CT block: [present, has_receivers, data_ready_sem_id, consumer_ready_sem_id, ack_count, flags, rotating_span]. Pass pre_handshake to override the flags bit for this emission.)doc")
        .def(
            "runtime_args",
            &kh::Mcast1D::runtime_args,
            nb::arg("core"),
            R"doc(Per-core runtime args. Fixed: 4 words (sender -> dest rect, receiver -> [sender_x, sender_y, 0, 0]); a fixed sender's rect is always the full receiver line; the device pipe excludes an in-line source. Rotating: 4 + 2*num_senders() words (full-line rect, then one sender coord pair per round).)doc")
        .def("is_sender", &kh::Mcast1D::is_sender, nb::arg("core"))
        .def("num_receivers", &kh::Mcast1D::num_receivers, nb::arg("core"))
        .def(
            "ack_count",
            &kh::Mcast1D::ack_count,
            R"doc(The sender's acknowledgment count, or MCAST_ACK_EQUALS_FANOUT when the kernel derives it per sender.)doc")
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
        .def(
            "has_receivers",
            &kh::Mcast1D::has_receivers,
            R"doc(True when at least one configured sender has a nonzero receiver fanout.)doc");

    // Mcast2D — one multicast over a receiver rectangle with an explicit fixed or rotating sender config.
    static_cast<nb::class_<kh::Mcast2D>>(mod.attr("Mcast2D"))
        .def(
            "__init__",
            [](kh::Mcast2D* self,
               MeshDevice* device,
               const CoreRangeSet& mcast_rect,
               const kh::Mcast2DSenderConfig& sender_config,
               const kh::McastConfig& config) { new (self) kh::Mcast2D(device, mcast_rect, sender_config, config); },
            nb::arg("device"),
            nb::arg("mcast_rect"),
            nb::arg("sender_config"),
            nb::arg("config") = kh::McastConfig{})
        .def(
            "owned_semaphores",
            &kh::Mcast2D::owned_semaphores,
            R"doc(The SemaphoreDescriptors this helper created, placed on the participating set (rect, or rect ∪ {sender}); empty if sem_ids were adopted.)doc")
        .def(
            "compile_time_args",
            &kh::Mcast2D::compile_time_args,
            nb::arg("pre_handshake") = std::optional<bool>{},
            R"doc(McastArgs CT block: [present, has_receivers, data_ready_sem_id, consumer_ready_sem_id, ack_count, flags, rotating_span]. Pass pre_handshake to override the flags bit for this emission.)doc")
        .def(
            "runtime_args",
            &kh::Mcast2D::runtime_args,
            nb::arg("core"),
            R"doc(Per-core runtime args. Fixed: 4 words (sender -> dest rect, receiver -> [sender_x, sender_y, 0, 0]). Rotating: 4 + 2*num_senders() words (full-rect rect, then one sender coord pair per round).)doc")
        .def("is_sender", &kh::Mcast2D::is_sender, nb::arg("core"))
        .def("num_receivers", &kh::Mcast2D::num_receivers, nb::arg("core"))
        .def(
            "ack_count",
            &kh::Mcast2D::ack_count,
            R"doc(The sender's acknowledgment count, or MCAST_ACK_EQUALS_FANOUT when the kernel derives it per sender.)doc")
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
        .def(
            "has_receivers",
            &kh::Mcast2D::has_receivers,
            R"doc(True when at least one configured sender has a nonzero receiver fanout.)doc");
}

}  // namespace ttnn::mcast_host
