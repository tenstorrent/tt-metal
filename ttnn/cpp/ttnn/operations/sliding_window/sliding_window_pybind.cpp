// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sliding_window_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "sliding_window.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::sliding_window {

void py_bind_sliding_window(py::module& module) {
    py::class_<ParallelConfig>(module, "ParallelConfig")
        .def(
            py::init<CoreRangeSet, TensorMemoryLayout, ShardOrientation>(),
            py::kw_only(),
            py::arg("grid"),
            py::arg("shard_scheme"),
            py::arg("shard_orientation"))
        .def_readwrite("grid", &ParallelConfig::grid)
        .def_readwrite("shard_scheme", &ParallelConfig::shard_scheme)
        .def_readwrite("shard_orientation", &ParallelConfig::shard_orientation);
}

}  // namespace ttnn::operations::sliding_window
