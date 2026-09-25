// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bindings for the pipelined-prefill layer-completion aggregation feature: the host-local ring
// (LayerCompletionQueue), the per-host router (LayerCompletionRouter), and the test-only scheduler
// stand-in consumer (LayerCompletionConsumer). Folded into the main ttnn module (`_ttnn`) as the
// `layer_completion` submodule via bind_layer_completion_api(); it was previously a standalone
// `_layer_completion` extension. The tt_metal types are consumed via the sanctioned
// `tt_metal/api/internal/disaggregation/` surface.
//
// The consumer is test-only and lives here rather than in tt_metal, so its binding is compiled only
// under TTNN_BUILD_TESTS (TTNN_WITH_LAYER_COMPLETION_CONSUMER) and is absent from a shipped wheel.
// The ring and router bindings are unconditional — they are production API.

#include "ttnn-nanobind/layer_completion.hpp"

#include <cstdint>
#include <optional>
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

#ifdef TTNN_WITH_LAYER_COMPLETION_CONSUMER
#include "ttnn-nanobind/layer_completion_consumer.hpp"
#endif

namespace ttnn::layer_completion {

namespace nb = nanobind;

void bind_layer_completion_api(nb::module_& mod) {
    using tt::tt_metal::internal::LayerCompletionMessage;
    using tt::tt_metal::internal::LayerCompletionMessageV2;
    using tt::tt_metal::internal::LayerCompletionProtocol;
    using tt::tt_metal::internal::LayerCompletionQueue;
    using tt::tt_metal::internal::LayerCompletionQueueBase;
    using tt::tt_metal::internal::LayerCompletionQueueV2;
    using tt::tt_metal::internal::LayerCompletionRouter;
    using tt::tt_metal::internal::LayerCompletionRouterConfig;

    mod.doc() = "Pipelined-prefill layer-completion ring/router/consumer.";

    // Protocol-neutral surface, shared by both ring versions. Everything a constructed ring
    // supports regardless of message version lives here; the versioned classes below add only
    // their create/connect factories and their message-typed try_push/try_pop.
    nb::class_<LayerCompletionQueueBase>(mod, "LayerCompletionQueueBase")
        .def(
            "shutdown",
            &LayerCompletionQueueBase::shutdown,
            "Idempotent teardown. Owner unlinks; connector unmaps.")
        .def_prop_ro("shm_name", &LayerCompletionQueueBase::shm_name)
        .def_prop_ro_static("capacity", [](nb::handle) { return LayerCompletionQueue::capacity(); });

    nb::class_<LayerCompletionQueue, LayerCompletionQueueBase>(mod, "LayerCompletionQueue")
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
            "Consumer pop. Returns (seq, source_rank, layer_idx, request_id) or None when empty.");

    nb::class_<LayerCompletionQueueV2, LayerCompletionQueueBase>(mod, "LayerCompletionQueueV2")
        .def_static(
            "create",
            &LayerCompletionQueueV2::create,
            nb::arg("shm_name"),
            "Create the v2 SHM ring as OWNER. Throws if the segment already exists.")
        .def_static(
            "connect",
            &LayerCompletionQueueV2::connect,
            nb::arg("shm_name"),
            nb::arg("connect_timeout_ms") = 30'000u,
            "Attach to an owner-created v2 ring by name (polls until present or timeout).")
        .def(
            "try_push",
            [](LayerCompletionQueueV2& self,
               uint64_t seq,
               uint32_t source_rank,
               uint32_t request_id,
               uint32_t slot_id,
               uint32_t pos_start,
               uint32_t pos_end,
               uint32_t layer_start,
               uint32_t layer_end) {
                return self.try_push(LayerCompletionMessageV2{
                    seq, source_rank, request_id, slot_id, pos_start, pos_end, layer_start, layer_end, 0u});
            },
            nb::arg("seq"),
            nb::arg("source_rank"),
            nb::arg("request_id"),
            nb::arg("slot_id"),
            nb::arg("pos_start"),
            nb::arg("pos_end"),
            nb::arg("layer_start"),
            nb::arg("layer_end"),
            "Producer push of a self-describing v2 completion (position + layer ranges). "
            "Returns False (no write) when the ring is full.")
        .def(
            "try_pop",
            [](LayerCompletionQueueV2& self)
                -> std::optional<
                    std::tuple<uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> {
                LayerCompletionMessageV2 m{};
                if (!self.try_pop(m)) {
                    return std::nullopt;
                }
                return std::make_tuple(
                    m.seq, m.source_rank, m.request_id, m.slot_id, m.pos_start, m.pos_end, m.layer_start,
                    m.layer_end);
            },
            "Consumer pop. Returns (seq, source_rank, request_id, slot_id, pos_start, pos_end, layer_start, "
            "layer_end) or None when empty.");

    nb::class_<LayerCompletionRouter>(mod, "LayerCompletionRouter")
        .def(
            "__init__",
            [](LayerCompletionRouter* self,
               int rank,
               int world_size,
               int master_rank,
               const std::string& ring_shm_name,
               const std::string& scheduler_shm_name,
               int poll_idle_us,
               int teardown_timeout_ms,
               int protocol) {
                LayerCompletionRouterConfig cfg;
                cfg.rank = rank;
                cfg.world_size = world_size;
                cfg.master_rank = master_rank;
                cfg.ring_shm_name = ring_shm_name;
                cfg.scheduler_shm_name = scheduler_shm_name;
                cfg.poll_idle_us = poll_idle_us;
                cfg.teardown_timeout_ms = teardown_timeout_ms;
                switch (protocol) {
                    case 1: cfg.protocol = LayerCompletionProtocol::kCountOnlyV1; break;
                    case 2: cfg.protocol = LayerCompletionProtocol::kStructuredV2; break;
                    default:
                        throw std::invalid_argument(
                            "LayerCompletionRouter: protocol must be 1 or 2, got " + std::to_string(protocol));
                }
                new (self) LayerCompletionRouter(std::move(cfg));
            },
            nb::arg("rank"),
            nb::arg("world_size"),
            nb::arg("master_rank"),
            nb::arg("ring_shm_name"),
            nb::arg("scheduler_shm_name") = std::string{},
            nb::arg("poll_idle_us") = 100,
            nb::arg("teardown_timeout_ms") = 5000,
            // Appended (after every pre-existing arg) so positional callers are unaffected.
            nb::arg("protocol") = 1,
            "Create the host's router: owns the local ring, spawns the listener thread, and on the master "
            "rank owns the scheduler-facing segment at scheduler_shm_name (one name for both protocols). "
            "protocol=1 (default): reorder to a bare count on a counter channel there. protocol=2: forward "
            "self-describing messages as-arrived into a structured ring there.")
        .def("stop", &LayerCompletionRouter::stop, "Idempotent: stop + join the listener thread.")
        .def_prop_ro("processed", &LayerCompletionRouter::processed)
        .def_prop_ro("is_master", &LayerCompletionRouter::is_master);

#ifdef TTNN_WITH_LAYER_COMPLETION_CONSUMER
    using tt::tests::prefill_test::LayerCompletionConsumer;

    nb::class_<LayerCompletionConsumer>(mod, "LayerCompletionConsumer")
        .def(
            "__init__",
            [](LayerCompletionConsumer* self,
               const std::string& channel_shm_name,
               uint64_t expected,
               uint32_t connect_timeout_ms,
               uint64_t log_step) {
                new (self) LayerCompletionConsumer(channel_shm_name, expected, connect_timeout_ms, log_step);
            },
            nb::arg("channel_shm_name"),
            nb::arg("expected"),
            nb::arg("connect_timeout_ms") = 30'000u,
            nb::arg("log_step") = 61u,
            "Test/scheduler stand-in: connect to the scheduler counter channel and drain it on a NATIVE "
            "C++ thread (GIL-immune), self-terminating once `expected` completions are drained.")
        .def("stop", &LayerCompletionConsumer::stop, "Idempotent: stop + join + final drain + shutdown channel.")
        .def_prop_ro("total", &LayerCompletionConsumer::total)
        .def_prop_ro("reached_expected", &LayerCompletionConsumer::reached_expected);
#endif
}

}  // namespace ttnn::layer_completion
