// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layer_ack_service.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ttnn/services/d2h_socket_service.hpp"
#include "ttnn/services/layer_ack_service.hpp"

namespace ttnn::layer_ack_service {

void py_module_types(nb::module_& mod) {
    using tt::tt_metal::D2HStreamService;
    using tt::tt_metal::LayerAckService;

    nb::class_<LayerAckService>(mod, "LayerAckService")
        .def(
            "__init__",
            [](LayerAckService* self,
               D2HStreamService& d2h_service,
               const std::string& ring_shm_name,
               uint32_t source_rank,
               uint32_t num_ack_layers,
               uint32_t ack_first_idx,
               uint32_t ack_local_count,
               uint32_t connect_timeout_ms,
               nb::sequence ack_layer_ids_seq,
               uint32_t protocol) {
                std::vector<uint32_t> ack_layer_ids;
                ack_layer_ids.reserve(nb::len(ack_layer_ids_seq));
                for (nb::handle h : ack_layer_ids_seq) {
                    ack_layer_ids.push_back(nb::cast<uint32_t>(h));
                }
                new (self) LayerAckService(
                    d2h_service,
                    ring_shm_name,
                    source_rank,
                    num_ack_layers,
                    ack_first_idx,
                    ack_local_count,
                    connect_timeout_ms,
                    std::move(ack_layer_ids),
                    protocol);
            },
            nb::arg("d2h_service"),
            nb::arg("ring_shm_name"),
            nb::arg("source_rank"),
            nb::arg("num_ack_layers"),
            nb::arg("ack_first_idx"),
            nb::arg("ack_local_count"),
            nb::arg("connect_timeout_ms") = 30'000u,
            nb::arg("ack_layer_ids") = nb::list(),
            nb::arg("protocol") = 1u,
            // LayerAckService holds a bare reference to d2h_service and must not
            // outlive it. Tie its Python lifetime to this object so it can't be
            // GC'd while the reader thread is still dereferencing it. The ring is
            // connected by name (start()), so no C++ queue object crosses to Python.
            nb::keep_alive<1, 2>(),
            R"doc(
                Bridge a metadata-only D2HStreamService to the host-local layer-completion ring.

                Args:
                    d2h_service: the ack records' D2H service; this object must not outlive it.
                    ring_shm_name (str): router-owned ring to connect to (leading '/', no other '/').
                    source_rank (int): this host's world rank.
                    num_ack_layers (int): ack records one chunk produces across all ranks (seq stride).
                    ack_first_idx (int): this rank's first ACK index.
                    ack_local_count (int): ack records this rank emits per chunk.
                    connect_timeout_ms (int): how long start() polls for the ring.
                    ack_layer_ids (list[int]): global layer of each local record, in emission
                        order; empty for a dense model (ack index == layer).
                    protocol (int): 1 counted, 2 structured; must match the router.
            )doc")
        .def(
            "start",
            &LayerAckService::start,
            R"doc(
                Launch the reader thread. Idempotent — calling start() again
                while already running is a no-op.
            )doc")
        .def(
            "stop",
            &LayerAckService::stop,
            nb::call_guard<nb::gil_scoped_release>(),
            "Stop and join the reader thread, then raise a stored failure once. Idempotent; call before the "
            "router stops.")
        .def(
            "check",
            &LayerAckService::check,
            nb::call_guard<nb::gil_scoped_release>(),
            "Raise a stored failure (lost D2H record, reader-thread error) without stopping.");
}

}  // namespace ttnn::layer_ack_service
