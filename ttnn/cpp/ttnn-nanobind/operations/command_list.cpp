// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_list.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/command_list.hpp"
#include "ttnn/common/queue_id.hpp"

namespace ttnn::operations::command_list {

using ttnn::experimental::CommandList;
using ttnn::experimental::CommandListBuilder;

void py_module_types(nb::module_& mod) {
    nb::class_<CommandListBuilder>(mod, "CommandListBuilder", R"doc(
        Records TTNN operations into command lists instead of running them (experimental).

        Inside ``with builder:``, every device operation launched on the calling thread is recorded, not executed.
        Outputs are allocated but hold stale data until a built list replays. Host reads and writes still run
        immediately and are not recorded. Ops on other threads run normally.

        Only Metal 2.0 (ProgramSpec) operations can be recorded; any other operation raises, naming the operation.
        No warm-up run is needed: recording compiles each workload and uploads its kernel binaries.

        Only one builder may exist per mesh device, and no trace can be captured while a builder or command list
        exists. An exception inside ``with builder:`` ends recording and keeps what was recorded so far; call
        ``clear()`` to discard it.

        Args:
            mesh_device (ttnn.MeshDevice): Mesh device to record on.
    )doc")
        .def(nb::init<tt::tt_metal::distributed::MeshDevice&>(), nb::arg("mesh_device"), nb::keep_alive<1, 2>())
        .def(
            "__enter__",
            [](CommandListBuilder& self) -> CommandListBuilder& {
                self.begin_recording();
                return self;
            },
            nb::rv_policy::reference)
        .def(
            "__exit__",
            [](CommandListBuilder& self, nb::handle, nb::handle, nb::handle) {
                self.end_recording();
                return false;
            },
            nb::arg("exc_type").none(),
            nb::arg("exc_value").none(),
            nb::arg("traceback").none())
        .def(
            "build",
            &CommandListBuilder::build,
            nb::kw_only(),
            nb::arg("cq_id") = nb::none(),
            nb::call_guard<nb::gil_scoped_release>(),
            R"doc(
                Builds an independent command list from everything recorded so far.

                Repeatable: each call allocates a new list in device DRAM. Built lists outlive the builder.

                Keyword Args:
                    cq_id (Optional[ttnn.QueueId]): Command queue the list replays on. Defaults to the current
                        command queue.

                Returns:
                    ttnn.experimental.CommandList
            )doc")
        .def(
            "clear",
            &CommandListBuilder::clear,
            nb::call_guard<nb::gil_scoped_release>(),
            "Drops everything recorded so far and keeps the device reservation. Not allowed while recording.")
        .def(
            "deallocate",
            &CommandListBuilder::deallocate,
            nb::call_guard<nb::gil_scoped_release>(),
            "Releases the device reservation and invalidates the builder. Not allowed while recording.")
        .def_prop_ro("is_recording", &CommandListBuilder::is_recording);

    nb::class_<CommandList>(mod, "CommandList", R"doc(
        A replayable command list in device DRAM, bound to the command queue it was built for (experimental).

        Lists record raw device addresses, not tensors. Replay reads and writes those addresses whether or not the
        tensors are still allocated; what lives there is the caller's responsibility. Released when garbage collected
        or by ``deallocate()``.
    )doc")
        .def(
            "enqueue",
            &CommandList::enqueue,
            nb::kw_only(),
            nb::arg("blocking") = true,
            nb::arg("check_dram_overlap") = false,
            nb::call_guard<nb::gil_scoped_release>(),
            R"doc(
                Replays the command list on the command queue it was built for.

                Keyword Args:
                    blocking (bool): Whether to wait for the replay to complete. Defaults to ``True``.
                    check_dram_overlap (bool): Before enqueueing, check every recorded tensor range against live DRAM
                        allocations. A free range passes. A range exactly filled by one allocation passes, with a
                        warning unless it is the bound tensor's own allocation (so a view always warns). Any other
                        overlap, including a view of part of an allocation, fails and nothing is enqueued. For
                        development; scans every live allocation. Defaults to ``False``.
            )doc")
        .def(
            "deallocate",
            &CommandList::deallocate,
            nb::call_guard<nb::gil_scoped_release>(),
            "Releases device resources and invalidates the list. Repeated calls have no effect.")
        .def_prop_ro("cq_id", &CommandList::cq_id);
}

}  // namespace ttnn::operations::command_list
