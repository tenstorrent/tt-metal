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
               uint32_t first_record,
               uint32_t record_count,
               uint32_t connect_timeout_ms,
               const nb::sequence& layer_ids_seq) {
                std::vector<uint32_t> layer_ids;
                layer_ids.reserve(nb::len(layer_ids_seq));
                for (nb::handle h : layer_ids_seq) {
                    layer_ids.push_back(nb::cast<uint32_t>(h));
                }
                new (self) LayerAckService(
                    d2h_service, ring_shm_name, first_record, record_count, connect_timeout_ms, std::move(layer_ids));
            },
            nb::arg("d2h_service"),
            nb::arg("ring_shm_name"),
            nb::arg("first_record"),
            nb::arg("record_count"),
            nb::arg("connect_timeout_ms") = 30'000u,
            nb::arg("layer_ids") = nb::list(),
            // The reader thread holds a bare reference to d2h_service: tie its Python lifetime to this object.
            nb::keep_alive<1, 2>(),
            R"doc(
                Bridge a metadata-only D2HStreamService to the host-local layer-completion ring.

                Args:
                    d2h_service: the ack records' D2H service; this object must not outlive it.
                    ring_shm_name (str): router-owned ring to connect to (leading '/', no other '/').
                    first_record (int): index of this rank's first record in a chunk's record sequence.
                    record_count (int): ack records this rank emits per chunk.
                    connect_timeout_ms (int): how long start() polls for the ring.
                    layer_ids (list[int]): global layer of each local record, in emission
                        order; empty for a dense model (record index == layer).
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
