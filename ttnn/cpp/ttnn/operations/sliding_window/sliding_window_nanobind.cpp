// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sliding_window_nanobind.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "sliding_window.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::sliding_window {

void bind_sliding_window(nb::module_& mod) {
    nb::class_<ParallelConfig>(mod, "ParallelConfig")
        .def(
            nb::init<CoreRangeSet, TensorMemoryLayout, ShardOrientation>(),
            nb::kw_only(),
            nb::arg("grid"),
            nb::arg("shard_scheme"),
            nb::arg("shard_orientation"))
        .def_rw("grid", &ParallelConfig::grid)
        .def_rw("shard_scheme", &ParallelConfig::shard_scheme)
        .def_rw("shard_orientation", &ParallelConfig::shard_orientation);
}

}  // namespace ttnn::operations::sliding_window
