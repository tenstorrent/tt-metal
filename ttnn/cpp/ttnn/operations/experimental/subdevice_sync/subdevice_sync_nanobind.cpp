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
#include "ttnn/operations/experimental/subdevice_sync/subdevice_sync.hpp"

namespace ttnn::operations::experimental::subdevice_sync::detail {

void bind_subdevice_sync_operations(nb::module_& mod) {
    mod.def(
        "subdevice_signal",
        [](const tt::tt_metal::GlobalSemaphore& semaphore,
           tt::tt_metal::SubDeviceId subdevice_id,
           const CoreCoord& core,
           const CoreCoord& consumer_core,
           uint32_t increment) {
            ttnn::operations::experimental::subdevice_signal(semaphore, subdevice_id, core, consumer_core, increment);
        },
        nb::arg("semaphore"),
        nb::kw_only(),
        nb::arg("subdevice_id"),
        nb::arg("core"),
        nb::arg("consumer_core"),
        nb::arg("increment") = 1,
        R"doc(
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
        )doc");

    mod.def(
        "subdevice_wait",
        [](const tt::tt_metal::GlobalSemaphore& semaphore,
           tt::tt_metal::SubDeviceId subdevice_id,
           const CoreCoord& core,
           uint32_t value,
           uint32_t timeout_cycles) {
            ttnn::operations::experimental::subdevice_wait(semaphore, subdevice_id, core, value, timeout_cycles);
        },
        nb::arg("semaphore"),
        nb::kw_only(),
        nb::arg("subdevice_id"),
        nb::arg("core"),
        nb::arg("value") = 1,
        nb::arg("timeout_cycles") = 0,
        R"doc(
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
        )doc");
}

}  // namespace ttnn::operations::experimental::subdevice_sync::detail
