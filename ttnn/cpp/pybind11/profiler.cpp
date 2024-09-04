// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "profiler.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace py = pybind11;

namespace ttnn {
namespace profiler {
namespace detail {
void ProfilerModule(py::module &m_profiler) {
    m_profiler.def("start_tracy_zone",&tt::tt_metal::op_profiler::start_tracy_zone,
            py::arg("source"), py::arg("functName"),py::arg("lineNum"), py::arg("color") = 0, R"doc(
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

    m_profiler.def("stop_tracy_zone",&tt::tt_metal::op_profiler::stop_tracy_zone, py::arg("name") = "", py::arg("color") = 0, R"doc(
        Stop profiling op with tracy.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | name             | Replace name for the zone                          | string                |             | No       |
        | color            | Replace zone color                             | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def(
        "tracy_message",
        &tt::tt_metal::op_profiler::tracy_message,
        py::arg("message"),
        py::arg("color") = 0xf0f8ff,
        R"doc(
        Emit a message signpost into the tracy profile.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | message          | Message description for this signpost.         | string                |             | Yes      |
        | color            | Zone color                                     | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def(
        "tracy_frame",
        &tt::tt_metal::op_profiler::tracy_frame,
        R"doc(
        Emit a tracy frame signpost.
    )doc");
}

}  // namespace detail

void py_module(py::module& module) {
   detail::ProfilerModule(module);
}
}  // namespace profiler
}  // namespace ttnn
