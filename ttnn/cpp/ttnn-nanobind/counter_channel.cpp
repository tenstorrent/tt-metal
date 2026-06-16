// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "counter_channel.hpp"

#include <cstdint>
#include <memory>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <internal/service/inter_process_counter_channel.hpp>

namespace ttnn::counter_channel {

void py_module_types(nb::module_& mod) {
    using tt::tt_metal::distributed::InterProcessCounterChannel;

    nb::class_<InterProcessCounterChannel>(mod, "InterProcessCounterChannel")
        .def(
            // ===== Owner-side ctor =====
            // Creates /dev/shm/<shm_name> via shm_open(O_CREAT|O_EXCL),
            // ftruncate, mmap. Stamps prior_clean_shutdown=1 so the
            // first connector to attach sees a clean prior. Throws
            // RuntimeError if a stale segment with the same name
            // already exists — caller must clean up before retrying.
            nb::init<const std::string&>(),
            nb::arg("shm_name"),
            R"doc(
                Create a new SHM-backed counter channel as the OWNER.

                The owner is the producer side: it creates the
                /dev/shm segment, holds its lifetime, and atomic-adds
                to the counter via `inject(n)`. A separate connector
                process attaches to the same segment via
                `InterProcessCounterChannel.connect(shm_name)` and
                drains events.

                Args:
                    shm_name (str): POSIX-shm name. Must start with a
                        single '/' and contain no other slashes.
                        Domain naming policy is the caller's
                        responsibility — pick a name unique to this
                        channel's role (e.g. layer-ack channel,
                        migration-resp channel) keyed by your
                        service id.

                Raises:
                    RuntimeError: A segment with this name already
                        exists. The owner is responsible for
                        unlinking a stale segment from a prior
                        crashed run (e.g. `os.unlink('/dev/shm' +
                        shm_name)`) before retrying.
            )doc")
        .def_static(
            // ===== Connector-side factory =====
            // Polls /dev/shm/<shm_name> until present (or timeout),
            // mmaps, reads + clears prior_clean_shutdown. Returns a
            // new connector-role instance; unique_ptr ownership
            // surrenders to Python.
            "connect",
            &InterProcessCounterChannel::connect,
            nb::arg("shm_name"),
            nb::arg("connect_timeout_ms") = 30'000u,
            R"doc(
                Attach to an owner-exported segment as the CONNECTOR.

                Polls /dev/shm/<shm_name> until the owner has
                exported it (or the timeout elapses), mmaps the
                segment, atomically reads + clears
                `prior_clean_shutdown` so subsequent connectors only
                see the immediately-previous one's exit state.

                Args:
                    shm_name (str): Same name the owner passed at
                        construction.
                    connect_timeout_ms (int, optional): How long to
                        wait for /dev/shm/<shm_name> to appear before
                        throwing. Default: 30 000 ms.

                Returns:
                    InterProcessCounterChannel: A new connector-role
                        instance.

                Raises:
                    RuntimeError: The owner did not export within
                        `connect_timeout_ms`, or another POSIX
                        failure occurred (e.g. permission denied).
            )doc")
        .def(
            // ===== Owner-only: inject =====
            "inject",
            &InterProcessCounterChannel::inject,
            nb::arg("n"),
            R"doc(
                Atomic-add `n` events to the channel's counter.

                OWNER-ONLY. Calling this on a connector instance
                raises RuntimeError.

                Args:
                    n (int): Number of events to publish. Typically
                        called once per produced event (n=1) but
                        batched values are also valid.
            )doc")
        .def(
            // ===== Connector-only: try_consume_all =====
            "try_consume_all",
            &InterProcessCounterChannel::try_consume_all,
            R"doc(
                Drain every event produced since the last call.

                CONNECTOR-ONLY. Returns 0 if nothing new since the
                last drain; otherwise returns the count and advances
                the consumer cursor. Non-blocking; safe to poll in a
                hot loop.

                Returns:
                    int: Number of events drained.
            )doc")
        .def(
            // ===== Connector-only: pending =====
            "pending",
            &InterProcessCounterChannel::pending,
            R"doc(
                Non-destructive snapshot of unconsumed events — i.e.
                what the next `try_consume_all()` would return.

                CONNECTOR-ONLY. Diagnostic only; the value can change
                between this call and the caller acting on it.

                Returns:
                    int: Currently unconsumed event count.
            )doc")
        .def(
            // ===== Connector-only diagnostic =====
            "had_clean_prior_shutdown",
            &InterProcessCounterChannel::had_clean_prior_shutdown,
            R"doc(
                Snapshot of `prior_clean_shutdown` taken at attach
                time (before this connector cleared it).

                CONNECTOR-ONLY. True iff the immediately-previous
                connector exited cleanly via `shutdown()` (i.e. its
                destructor ran to completion). False on first attach
                to a fresh segment, or after a connector that
                crashed without running its dtor. Informational —
                the SHM imposes no recovery policy.

                Returns:
                    bool: True if the previous connector exited
                        cleanly.
            )doc")
        .def(
            // ===== Both roles: explicit shutdown =====
            // Idempotent. The Python wrapper's __del__ will also
            // call this via the C++ dtor, but explicit teardown is
            // preferable for predictable /dev/shm lifecycle (Python
            // GC timing isn't deterministic).
            "shutdown",
            &InterProcessCounterChannel::shutdown,
            R"doc(
                Tear down the channel. Idempotent.

                Behaviour is role-dispatched:
                  * Owner    : munmap + shm_unlink. The segment is
                               removed from /dev/shm; any
                               still-attached connector keeps its
                               mapping until it itself unmaps, but
                               no new connector can find the name.
                  * Connector: stamps prior_clean_shutdown=1 in the
                               segment, then munmaps. The segment
                               itself survives (the owner side owns
                               unlinking), and the next connector
                               sees a clean prior.

                Subsequent calls on either role are no-ops. The
                destructor calls this if the caller didn't.
            )doc")
        .def_prop_ro(
            // ===== Both roles: read-only property =====
            "shm_name",
            &InterProcessCounterChannel::shm_name,
            R"doc(
                The /dev/shm-relative name the channel was created or
                connected with. Read-only.
            )doc");
}

void py_module(nb::module_& /* mod */) {
    // No free functions; the channel is exposed entirely via
    // py_module_types.
}

}  // namespace ttnn::counter_channel
