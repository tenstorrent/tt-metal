// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layer_ack_service.hpp"

#include <nanobind/nanobind.h>

#include <internal/service/inter_process_counter_channel.hpp>

#include "ttnn/services/d2h_socket_service.hpp"
#include "ttnn/services/layer_ack_service.hpp"

namespace ttnn::layer_ack_service {

void py_module_types(nb::module_& mod) {
    using tt::tt_metal::D2HStreamService;
    using tt::tt_metal::LayerAckService;
    using tt::tt_metal::distributed::InterProcessCounterChannel;

    nb::class_<LayerAckService>(mod, "LayerAckService")
        .def(
            nb::init<D2HStreamService&, InterProcessCounterChannel&>(),
            nb::arg("d2h_service"),
            nb::arg("ack_channel"),
            // LayerAckService holds bare references and must not outlive either
            // input. Tie their Python lifetimes to this object so neither can be
            // GC'd while the reader thread is still dereferencing them.
            nb::keep_alive<1, 2>(),
            nb::keep_alive<1, 3>(),
            R"doc(
                Bridge a metadata-only D2HStreamService to a scheduler-facing
                InterProcessCounterChannel.

                Runs an internal reader thread (see start()) that blocks on
                d2h_service.read_metadata() — one record per completed layer —
                and calls ack_channel.inject(1) per record. Does NOT own or
                construct either argument; both must outlive this service.

                Args:
                    d2h_service (D2HStreamService): A service constructed in
                        metadata-only mode (global_spec=None,
                        metadata_size_bytes > 0).
                    ack_channel (InterProcessCounterChannel): The owner-side
                        (producer) ack channel the scheduler drains.
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

                NOTE: read_metadata() blocks until the device sends the next
                record, so stop() blocks on the join until a record arrives (or
                the parked read returns). The destructor also calls stop().
            )doc");
}

void py_module(nb::module_& /* mod */) {
    // No free functions; the service is exposed entirely via py_module_types.
}

}  // namespace ttnn::layer_ack_service
