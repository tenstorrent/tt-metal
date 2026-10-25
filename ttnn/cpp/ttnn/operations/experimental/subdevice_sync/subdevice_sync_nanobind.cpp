// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subdevice_sync_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/subdevice_sync/subdevice_sync.hpp"

namespace ttnn::operations::experimental::subdevice_sync::detail {

namespace {
constexpr const char* SIGNAL_DOC = R"doc(
            In-trace cross-sub-device hand-off, producer side. Enqueues a one-core program on every chip of the
            semaphore's mesh, on ``core`` of the PRODUCER's sub-device ``subdevice_id``. Its go signal is sent only
            after every earlier program on ``subdevice_id`` has completed on all its cores; it then atomically adds
            ``increment`` to ``semaphore`` in the L1 of ``consumer_core`` and exits. Issue it right after the
            producer op(s). Semaphore address / consumer_core / increment are runtime args (one cached program per
            (core, sub-device)).

            Args:
                semaphore (ttnn.GlobalSemaphore): ttnn.create_global_semaphore(mesh, <consumer_core range>, 0).
                subdevice_id (ttnn.SubDeviceId): the producer's sub-device; must contain ``core``.
                core (ttnn.CoreCoord): logical core that runs the signal kernel.
                consumer_core (ttnn.CoreCoord): logical core holding the semaphore (the subdevice_wait core).
                increment (int): default 1.
        )doc";

constexpr const char* WAIT_DOC = R"doc(
            In-trace cross-sub-device hand-off, consumer side. Enqueues a one-core program on every chip, on ``core``
            of the CONSUMER's sub-device ``subdevice_id``: it spins until ``semaphore`` (in its own L1) is >=
            ``value``, then atomically SUBTRACTS ``value`` (counting semaphore; increments that race with the
            consume are never lost and the semaphore is back at its initial value after every balanced pair, so
            each trace replay starts clean). The next program on ``subdevice_id`` starts only after it completes.

            Issue-order rule (single CQ): enqueue the matching subdevice_signal BEFORE the next program on
            ``subdevice_id`` after this wait, or the CQ deadlocks (go signals are released in issue order and that
            program's go waits for this wait). Note that within one CQ that same rule already orders the consumer
            after the producer (the signal's go waits for the producer), so the wait is a belt-and-braces guard;
            it is load-bearing across command queues.

            Args:
                semaphore (ttnn.GlobalSemaphore)
                subdevice_id (ttnn.SubDeviceId): the consumer's sub-device; must contain ``core``.
                core (ttnn.CoreCoord): logical core holding the semaphore word.
                value (int): count to wait for and consume (default 1; e.g. 2 for two producers).
                timeout_cycles (int): DEBUG ONLY, 0 = wait forever. On timeout the program exits without consuming
                    (the consumer then runs unordered): turns a would-be hang into detectable violations.
        )doc";

constexpr const char* FENCE_DOC = R"doc(
            Single-CQ cross-sub-device ordering without a semaphore. Enqueues an empty one-core program on
            ``subdevice_id`` (on every chip). Its go signal waits for every earlier program on ``subdevice_id``, and
            every program issued after it on the same CQ (any sub-device) waits behind its go. So
            ``producer(SD0); subdevice_fence(subdevice_id=SD0); consumer(SD1)`` orders the consumer after the producer.
            Not needed when the producer stream already issues another SD0 op before the consumer.

            Args:
                mesh_device (ttnn.MeshDevice)
                subdevice_id (ttnn.SubDeviceId): the PRODUCER's sub-device.
                core (ttnn.CoreCoord, optional): default first worker core of the sub-device.
        )doc";

}  // namespace

void bind_subdevice_sync_operations(nb::module_& mod) {
    ttnn::bind_function<"subdevice_signal", "ttnn.experimental.">(
        mod,
        SIGNAL_DOC,
        &ttnn::operations::experimental::subdevice_signal,
        nb::arg("semaphore"),
        nb::kw_only(),
        nb::arg("subdevice_id"),
        nb::arg("core"),
        nb::arg("consumer_core"),
        nb::arg("increment") = 1);

    ttnn::bind_function<"subdevice_wait", "ttnn.experimental.">(
        mod,
        WAIT_DOC,
        &ttnn::operations::experimental::subdevice_wait,
        nb::arg("semaphore"),
        nb::kw_only(),
        nb::arg("subdevice_id"),
        nb::arg("core"),
        nb::arg("value") = 1,
        nb::arg("timeout_cycles") = 0);

    ttnn::bind_function<"subdevice_fence", "ttnn.experimental.">(
        mod,
        FENCE_DOC,
        &ttnn::operations::experimental::subdevice_fence,
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("subdevice_id"),
        nb::arg("core") = nb::none());
}

}  // namespace ttnn::operations::experimental::subdevice_sync::detail
