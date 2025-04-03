// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include "sliding_window.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
enum class ShardOrientation;
enum class TensorMemoryLayout;
}  // namespace tt_metal
}  // namespace tt

using namespace tt::tt_metal;

namespace py = pybind11;
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
