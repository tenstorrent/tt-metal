// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sliding_window_nanobind.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "sliding_window.hpp"
#include "op_slicing/op_slicing.hpp"

using namespace tt::tt_metal;
using namespace ttnn::operations::op_slicing;

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

    auto py_op_slice_config = nb::class_<Op2DSliceConfig>(
        mod,
        "Op2DSliceConfig",
        R"doc(
        | Op2DSliceConfig is a structure that is used to configure how the input & output tensors of Conv2D are sliced when they are placed in DRAM. \
        | Conv2D only supports inputs in L1. If the input tensor or output tensor are too large to fit into L1, then the Conv2d_DRAM version can be used. \
        | It slices the input & output into slices and applies the relevant L1 op on each slice. \
        | Op2DSliceConfig determines how this slicing happens.
        )doc");
    py_op_slice_config.def(
        nb::init<Op2DSliceConfig::SliceType, uint32_t>(), nb::kw_only(), nb::arg("slice_type"), nb::arg("num_slices"));
    py_op_slice_config.def(nb::init<Op2DSliceConfig::SliceType>(), nb::kw_only(), nb::arg("slice_type"));
    py_op_slice_config.def("__repr__", [](const Op2DSliceConfig& config) { return fmt::format("{}", config); });
    py_op_slice_config.def_rw(
        "slice_type",
        &Op2DSliceConfig::slice_type,
        R"doc(
        | The type of slice to be used. Can be either SliceHeight or SliceWidth. When the tensor is in [N, H, W, C] format, then it can slice either along the height or width dimension.
        | Slicing along the width is preferable as it reduces the size of the output of the Halo operation.
        | Use SliceHeight only when the height dimension is much larger than the width dimension.
        )doc");
    py_op_slice_config.def_rw(
        "num_slices",
        &Op2DSliceConfig::num_slices,
        R"doc(
        | The number of slices that the input & output tensors are divided into.
        | The output tensor is divided into num_slices slices along the slice_type dimension.
        | The corresponding input tensor needed to calculate that output is determined and sliced.
        | If the size of the slice dimension is not divisible by num_slices, then the last slice will be smaller than the rest.
        )doc");
    nb::enum_<Op2DSliceConfig::SliceType>(py_op_slice_config, "SliceTypeEnum")
        .value("L1Full", Op2DSliceConfig::SliceType::L1_FULL)
        .value("DRAMSliceHeight", Op2DSliceConfig::SliceType::DRAM_HEIGHT)
        .value("DRAMSliceWidth", Op2DSliceConfig::SliceType::DRAM_WIDTH);
}

}  // namespace ttnn::operations::sliding_window
