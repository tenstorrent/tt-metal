// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "op_slicing.hpp"
#include "op_slicing_pybind.hpp"
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include "ttnn-pybind/decorators.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::op_slicing {

void py_bind_op_slicing(py::module& module) {
    module.def(
        "run_sliced_op",
        py::overload_cast<const ttnn::Tensor&, ttnn::Tensor&, const std::vector<OpSliceAttr*>&, Op2DSliceConfig>(
            &run_sliced_op),
        py::kw_only(),
        py::arg("input_tensor"),
        py::arg("output_tensor"),
        py::arg("op_slice_attr"),
        py::arg("dram_slice_config"),
        R"doc(
            This function is used as a performance optimization for sliced ops using DRAM Inputs & Outputs. This function allows for fusion of multiple OPs to execute on a single DRAM Input slice, and have only the output slice of the final op is written back to DRAM.
            This reduces the impact of the relatively slow DRAM access by amortizing it across multiple operations.

        Keyword Args:
            input_tensor (ttnn.Tensor):  The input to the first op. This tensor must be DRAM Interleaved, with it's a 4D Shape.
            output_tensor (ttnn.Tensor):  The tensor to which the output of the final op is written to. This tensor must also be DRAM Interleaved, with a 4D Shape.
            op_slice_attr (list[ttnn.OpSliceAttr]) : List describing the ops to be executed by passing objects of the operation's specific slicing attributes. For example, a list containing Conv2dSliceAttr and Pool2dSliceAttr.
            dram_slice_config (ttnn.Op2DSliceConfig) : Object that describes how the slicing should take place, which includes the dimension being sliced (currently only Height & Width are supported) & the number of slices.
    )doc");
    module.def(
        "run_sliced_op",
        py::overload_cast<const ttnn::Tensor&, const std::vector<OpSliceAttr*>&, Op2DSliceConfig>(&run_sliced_op),
        py::kw_only(),
        py::arg("input_tensor"),
        py::arg("op_slice_attr"),
        py::arg("dram_slice_config"),
        R"doc(
            This function is used as a performance optimization for sliced ops using DRAM Inputs & Outputs. This function allows for fusion of multiple OPs to execute on a single DRAM Input slice, and have only the output slice of the final op is written back to DRAM.
            This reduces the impact of the relatively slow DRAM access by amortizing it across multiple operations.

        Keyword Args:
            input_tensor (ttnn.Tensor):  The input to the first op. This tensor must be DRAM Interleaved, with it's a 4D Shape.
            op_slice_attr (list[ttnn.OpSliceAttr]) : List describing the ops to be executed by passing objects of the operation's specific slicing attributes. For example, a list containing Conv2dSliceAttr and Pool2dSliceAttr.
            dram_slice_config (ttnn.Op2DSliceConfig) : Object that describes how the slicing should take place, which includes the dimension being sliced (currently only Height & Width are supported) & the number of slices.

        Returns:
            The output tensor of the final op
            - ttnn.Tensor: The output tensor
        )doc");

    py::class_<OpSliceAttr> py_op_slice_attr(
        module,
        "OpSliceAttr",
        R"doc(
        OpSliceAttr is an interface that defines how to slice the input tensor based on the output slice.
        )doc");
}

}  // namespace ttnn::operations::op_slicing
