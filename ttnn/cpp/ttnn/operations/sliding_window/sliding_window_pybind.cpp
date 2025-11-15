// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sliding_window_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "sliding_window.hpp"
#include "op_slicing/op_slicing.hpp"
using namespace tt::tt_metal;
using namespace ttnn::operations::op_slicing;
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

    auto py_op_slice_config = py::class_<Op2dSliceConfig>(
        module,
        "Op2dSliceConfig",
        R"doc(
        | Op2dSliceConfig is a structure that is used to configure how the input & output tensors of Conv2D are sliced when they are placed in DRAM. \
        | Conv2D only supports inputs in L1. If the input tensor or output tensor are too large to fit into L1, then the Conv2d_DRAM version can be used. \
        | It slices the input & output into slices and applies the relevant L1 op on each slice. \
        | Op2dSliceConfig determines how this slicing happens.
        )doc");
    py_op_slice_config.def(
        py::init<Op2dSliceConfig::SliceType, uint32_t>(), py::kw_only(), py::arg("slice_type"), py::arg("num_slices"));
    py_op_slice_config.def(py::init<Op2dSliceConfig::SliceType>(), py::kw_only(), py::arg("slice_type"));
    py_op_slice_config.def("__repr__", [](const Op2dSliceConfig& config) { return fmt::format("{}", config); });
    py_op_slice_config.def_readwrite(
        "slice_type",
        &Op2dSliceConfig::slice_type,
        R"doc(
        | The type of slice to be used. Can be either SliceHeight or SliceWidth. When the tensor is in [N, H, W, C] format, then it can slice either along the height or width dimension.
        | Slicing along the width is preferable as it reduces the size of the output of the Halo operation.
        | Use SliceHeight only when the height dimension is much larger than the width dimension.
        )doc");
    py_op_slice_config.def_readwrite(
        "num_slices",
        &Op2dSliceConfig::num_slices,
        R"doc(
        | The number of slices that the input & output tensors are divided into.
        | The output tensor is divided into num_slices slices along the slice_type dimension.
        | The corresponding input tensor needed to calculate that output is determined and sliced.
        | If the size of the slice dimension is not divisible by num_slices, then the last slice will be smaller than the rest.
        )doc");
    py::enum_<Op2dSliceConfig::SliceType>(py_op_slice_config, "SliceTypeEnum")
        .value("L1Full", Op2dSliceConfig::SliceType::L1_FULL)
        .value("DRAMSliceHeight", Op2dSliceConfig::SliceType::DRAM_HEIGHT)
        .value("DRAMSliceWidth", Op2dSliceConfig::SliceType::DRAM_WIDTH);
}

}  // namespace ttnn::operations::sliding_window
