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
            nb::init<
                D2HStreamService&,
                const std::string&,
                uint32_t,
                uint32_t,
                uint32_t,
                uint32_t,
                uint32_t,
                std::vector<uint32_t>,
                uint32_t>(),
            nb::arg("d2h_service"),
            nb::arg("ring_shm_name"),
            nb::arg("source_rank"),
            nb::arg("num_layers"),
            nb::arg("first_layer_idx"),
            nb::arg("local_layers"),
            nb::arg("connect_timeout_ms") = 30'000u,
            nb::arg("ack_layer_ids") = std::vector<uint32_t>{},
            nb::arg("protocol") = 1u,
            // LayerAckService holds a bare reference to d2h_service and must not
            // outlive it. Tie its Python lifetime to this object so it can't be
            // GC'd while the reader thread is still dereferencing it. The ring is
            // connected by name (start()), so no C++ queue object crosses to Python.
            nb::keep_alive<1, 2>(),
            R"doc(
                Bridge a metadata-only D2HStreamService to the pipelined-prefill
                layer-completion ring (LayerCompletionQueue).

                Runs an internal reader thread (see start()) that drains
                d2h_service.read_metadata() — one record per completed layer —
                and, for each record, derives a globally-dense ordering key and
                pushes one completion message into the router-owned ring. Pure
                producer: it never touches the scheduler counter channel (the
                LayerCompletionRouter owns aggregation + injection). Does NOT own
                d2h_service, which must outlive this service.

                Args:
                    d2h_service (D2HStreamService): A service constructed in
                        metadata-only mode (global_spec=None,
                        metadata_size_bytes > 0).
                    ring_shm_name (str): Name of the router-owned LayerCompletionQueue
                        ring to connect to (leading '/', no other slashes). The
                        LayerCompletionRouter must be constructed first.
                    source_rank (int): This host's world rank (stamped on each message).
                    num_layers (int): Global total ACK-record count — the seq stride.
                    first_layer_idx (int): This rank's ACK-space slice offset (the number
                        of ACK records all lower ranks emit per chunk).
                    local_layers (int): ACK records this rank emits per chunk. On a hybrid
                        stack that acks only on KV-writing layers this is the KV-slab
                        count, not the layer count.
                    connect_timeout_ms (int): Max time start() polls for the ring.
                    ack_layer_ids (list[int]): GLOBAL layer index of each of this rank's
                        ACK records, in emission order. Empty (default) = dense model,
                        where the ack index already IS the global layer. Supplying it is
                        what keeps `layer_idx` on the wire meaning the same thing as it
                        does on the host-callback transport.
                    protocol (int): 1 = v1 count protocol; 2 = v2 structured protocol,
                        which carries the record's real {slot_id, actual_start,
                        actual_end} instead of discarding it. Must match the router.
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
            R"doc(
                Signal the reader thread to exit and join it. Idempotent.

                The reader polls has_data() before each read, so it never parks
                inside a blocking read; stop() joins the thread promptly even when
                no further record arrives. The destructor also calls stop().
            )doc");
}

}  // namespace ttnn::layer_ack_service
