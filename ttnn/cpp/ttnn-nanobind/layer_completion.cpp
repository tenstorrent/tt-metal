// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layer_completion.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>

#include <internal/service/layer_completion_message.hpp>
#include <internal/service/layer_completion_queue.hpp>
#include <internal/service/layer_completion_router.hpp>

namespace ttnn::layer_completion {

void py_module_types(nb::module_& mod) {
    using tt::tt_metal::distributed::LayerCompletionMessage;
    using tt::tt_metal::distributed::LayerCompletionQueue;
    using tt::tt_metal::distributed::LayerCompletionRouter;
    using tt::tt_metal::distributed::LayerCompletionRouterConfig;

    nb::class_<LayerCompletionQueue>(mod, "LayerCompletionQueue")
        .def_static(
            "create",
            &LayerCompletionQueue::create,
            nb::arg("shm_name"),
            "Create the host-local SHM ring as OWNER. Throws if the segment already exists.")
        .def_static(
            "connect",
            &LayerCompletionQueue::connect,
            nb::arg("shm_name"),
            nb::arg("connect_timeout_ms") = 30'000u,
            "Attach to an owner-created ring by name (polls until present or timeout).")
        .def(
            "try_push",
            [](LayerCompletionQueue& self,
               uint64_t seq,
               uint32_t source_rank,
               uint32_t layer_idx,
               uint32_t request_id) {
                return self.try_push(LayerCompletionMessage{seq, source_rank, layer_idx, request_id, 0u});
            },
            nb::arg("seq"),
            nb::arg("source_rank"),
            nb::arg("layer_idx"),
            nb::arg("request_id"),
            "Producer push. Returns False (no write) when the ring is full.")
        .def(
            "try_pop",
            [](LayerCompletionQueue& self) -> std::optional<std::tuple<uint64_t, uint32_t, uint32_t, uint32_t>> {
                LayerCompletionMessage m{};
                if (!self.try_pop(m)) {
                    return std::nullopt;
                }
                return std::make_tuple(m.seq, m.source_rank, m.layer_idx, m.request_id);
            },
            "Consumer pop. Returns (seq, source_rank, layer_idx, request_id) or None when empty.")
        .def("shutdown", &LayerCompletionQueue::shutdown, "Idempotent teardown. Owner unlinks; connector unmaps.")
        .def_prop_ro("shm_name", &LayerCompletionQueue::shm_name)
        .def_prop_ro_static("capacity", [](nb::handle) { return LayerCompletionQueue::capacity(); });

    nb::class_<LayerCompletionRouter>(mod, "LayerCompletionRouter")
        .def(
            "__init__",
            [](LayerCompletionRouter* self,
               int rank,
               int world_size,
               int master_rank,
               const std::string& ring_shm_name,
               const std::string& scheduler_channel_shm_name,
               int poll_idle_us) {
                LayerCompletionRouterConfig cfg;
                cfg.rank = rank;
                cfg.world_size = world_size;
                cfg.master_rank = master_rank;
                cfg.ring_shm_name = ring_shm_name;
                cfg.scheduler_channel_shm_name = scheduler_channel_shm_name;
                cfg.poll_idle_us = poll_idle_us;
                new (self) LayerCompletionRouter(std::move(cfg));
            },
            nb::arg("rank"),
            nb::arg("world_size"),
            nb::arg("master_rank"),
            nb::arg("ring_shm_name"),
            nb::arg("scheduler_channel_shm_name") = std::string{},
            nb::arg("poll_idle_us") = 100,
            "Create the host's router: owns the local ring, spawns the listener thread, and on the master "
            "rank owns the scheduler-facing counter channel.")
        .def("stop", &LayerCompletionRouter::stop, "Idempotent: stop + join the listener thread.")
        .def_prop_ro("processed", &LayerCompletionRouter::processed)
        .def_prop_ro("is_master", &LayerCompletionRouter::is_master);
}

void py_module(nb::module_& /* mod */) {}

}  // namespace ttnn::layer_completion
