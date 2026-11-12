// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bindings for the layer-completion ring and router: the `layer_completion` submodule of `_ttnn`.

#include "ttnn-nanobind/layer_completion.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>

#include <internal/disaggregation/layer_completion_message.hpp>
#include <internal/disaggregation/layer_completion_queue.hpp>
#include <internal/disaggregation/layer_completion_router.hpp>

namespace ttnn::layer_completion {

namespace nb = nanobind;

void bind_layer_completion_api(nb::module_& mod) {
    using tt::tt_metal::internal::LayerCompletionMessage;
    using tt::tt_metal::internal::LayerCompletionQueue;
    using tt::tt_metal::internal::LayerCompletionRouter;
    using tt::tt_metal::internal::LayerCompletionRouterConfig;

    mod.doc() = "Pipelined-prefill layer-completion ring/router.";

    nb::class_<LayerCompletionQueue>(mod, "LayerCompletionQueue")
        .def_static(
            "create",
            &LayerCompletionQueue::create,
            nb::arg("shm_name"),
            nb::arg("completions_per_chunk"),
            "Create the SHM ring as OWNER. Throws if the segment already exists.")
        .def_static(
            "connect",
            &LayerCompletionQueue::connect,
            nb::arg("shm_name"),
            nb::arg("connect_timeout_ms") = 30'000u,
            "Attach to an owner-created ring by name (polls until present or timeout).")
        .def(
            "try_push",
            [](LayerCompletionQueue& self,
               uint32_t slot_id,
               uint32_t pos_start,
               uint32_t pos_end,
               uint32_t layer_start,
               uint32_t layer_end) {
                return self.try_push(LayerCompletionMessage{
                    slot_id,
                    pos_start,
                    pos_end,
                    layer_start,
                    layer_end,
                    0u,
                    tt::tt_metal::internal::layer_completion_host_timestamp_ns()});
            },
            nb::arg("slot_id"),
            nb::arg("pos_start"),
            nb::arg("pos_end"),
            nb::arg("layer_start"),
            nb::arg("layer_end"),
            "Producer push, stamped with the host time. Returns False (no write) when the ring is full.")
        .def(
            "try_pop",
            [](LayerCompletionQueue& self)
                -> std::optional<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t>> {
                LayerCompletionMessage m{};
                if (!self.try_pop(m)) {
                    return std::nullopt;
                }
                return std::make_tuple(
                    m.slot_id, m.pos_start, m.pos_end, m.layer_start, m.layer_end, m.host_timestamp_ns);
            },
            "Consumer pop. Returns (slot_id, pos_start, pos_end, layer_start, layer_end, host_timestamp_ns) or None "
            "when "
            "empty.")
        .def("shutdown", &LayerCompletionQueue::shutdown, "Idempotent teardown. Owner unlinks; connector unmaps.")
        .def_prop_ro("shm_name", &LayerCompletionQueue::shm_name)
        .def_prop_ro("completions_per_chunk", &LayerCompletionQueue::completions_per_chunk);

    nb::class_<LayerCompletionRouter>(mod, "LayerCompletionRouter")
        .def(
            "__init__",
            [](LayerCompletionRouter* self,
               int rank,
               int world_size,
               int master_rank,
               const std::string& ring_shm_name,
               const std::string& scheduler_shm_name,
               int completions_per_chunk,
               int poll_idle_us,
               int teardown_timeout_ms) {
                LayerCompletionRouterConfig cfg;
                cfg.rank = rank;
                cfg.world_size = world_size;
                cfg.master_rank = master_rank;
                cfg.ring_shm_name = ring_shm_name;
                cfg.scheduler_shm_name = scheduler_shm_name;
                if (completions_per_chunk <= 0) {
                    throw std::invalid_argument("LayerCompletionRouter: completions_per_chunk must be > 0");
                }
                cfg.completions_per_chunk = static_cast<uint32_t>(completions_per_chunk);
                cfg.poll_idle_us = poll_idle_us;
                cfg.teardown_timeout_ms = teardown_timeout_ms;
                new (self) LayerCompletionRouter(std::move(cfg));
            },
            nb::arg("rank"),
            nb::arg("world_size"),
            nb::arg("master_rank"),
            nb::arg("ring_shm_name"),
            nb::arg("scheduler_shm_name") = std::string{},
            nb::arg("completions_per_chunk"),
            nb::arg("poll_idle_us") = 100,
            nb::arg("teardown_timeout_ms") = 5000,
            "Create the host's router: owns the local ring, spawns the listener thread, and on the master "
            "rank owns the scheduler-facing ring at scheduler_shm_name, forwarding self-describing messages "
            "as they arrive.")
        .def(
            "stop",
            &LayerCompletionRouter::stop,
            nb::call_guard<nb::gil_scoped_release>(),
            "Idempotent: stop and join the listener thread, then raise a listener failure once.")
        .def_prop_ro("processed", &LayerCompletionRouter::processed)
        .def_prop_ro("is_master", &LayerCompletionRouter::is_master);
}

}  // namespace ttnn::layer_completion
