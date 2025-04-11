// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "profiler.hpp"

#include <nanobind/nanobind.h>

#include "tools/profiler/op_profiler.hpp"

namespace ttnn::profiler {

namespace {
void ProfilerModule(nb::module_& mod) {
    mod.def(
        "start_tracy_zone",
        &tt::tt_metal::op_profiler::start_tracy_zone,
        nb::arg("source"),
        nb::arg("functName"),
        nb::arg("lineNum"),
        nb::arg("color") = 0,
        R"doc(
        Stop profiling op with tracy.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | source           | Source file for the zone                       | string                |             | Yes      |
        | functName        | Function of the zone                           | string                |             | Yes      |
        | lineNum          | Line number of the zone marker                 | int                   |             | Yes      |
        | color            | Zone color                                     | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    mod.def(
        "stop_tracy_zone",
        &tt::tt_metal::op_profiler::stop_tracy_zone,
        nb::arg("name") = "",
        nb::arg("color") = 0,
        R"doc(
        Stop profiling op with tracy.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | name             | Replace name for the zone                          | string                |             | No       |
        | color            | Replace zone color                             | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    mod.def(
        "tracy_message",
        &tt::tt_metal::op_profiler::tracy_message,
        nb::arg("message"),
        nb::arg("color") = 0xf0f8ff,
        R"doc(
        Emit a message signpost into the tracy profile.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | message          | Message description for this signpost.         | string                |             | Yes      |
        | color            | Zone color                                     | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    mod.def(
        "tracy_frame",
        &tt::tt_metal::op_profiler::tracy_frame,
        R"doc(
        Emit a tracy frame signpost.
    )doc");
}

}  // namespace

void py_module(nb::module_& mod) { ProfilerModule(mod); }

}  // namespace ttnn::profiler
