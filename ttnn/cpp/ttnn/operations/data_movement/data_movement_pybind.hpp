// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/data_movement.hpp"
#include "ttnn/operations/data_movement/concat/concat_pybind.hpp"
#include "ttnn/operations/data_movement/pad/pad_pybind.hpp"
#include "ttnn/operations/data_movement/permute/permute_pybind.hpp"
#include "ttnn/operations/data_movement/downsample/downsample_op_pybind.hpp"
#include "ttnn/operations/data_movement/slice/slice_pybind.hpp"
#include "ttnn/operations/data_movement/tilize/tilize_pybind.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding_pybind.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave_pybind.hpp"
#include "ttnn/operations/data_movement/transpose/transpose_pybind.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace data_movement {

void bind_upsample(py::module& module) {
    const auto doc = R"doc(
 Upsamples a given multi-channel 2D (spatial) data.
 The input data is assumed to be of the form [N, H, W, C].

 The algorithms available for upsampling are 'nearest' for now.

 Args:
     * :attr:`input_tensor`: the input tensor
     * :attr:`scale_factor`: multiplier for spatial size. Has to match input size if it is a tuple.
     )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::upsample,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"), py::arg("scale_factor"), py::arg("memory_config") = std::nullopt});
}

void bind_repeat(py::module& module) {
    auto doc = R"doc(
repeat(input_tensor: ttnn.Tensor, shape : ttnn.Shape) -> ttnn.Tensor

Returns a new tensor filled with repetition of input :attr:`input_tensor` according to number of times specified in :attr:`shape`.

Args:
    * :attr:`input_tensor`: the input_tensor to apply the repeate operation.
    * :attr:`shape`: The number of repetitions for each element.

Keyword Args:
    * :attr:`memory_config`: the memory configuration to use for the operation

Example:

    >>> tensor = ttnn.repeat(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]]), 2,)), device)
    >>> print(tensor)
    tensor([[1, 2],
    [1, 2],
    [3, 4],
    [3, 4]])
        )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::repeat,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"), py::arg("shape"), py::kw_only(), py::arg("memory_config") = std::nullopt});
}

void py_module(py::module& module) {
    detail::bind_permute(module);
    detail::bind_concat(module);
    bind_upsample(module);
    detail::bind_pad(module);
    detail::bind_slice(module);
    detail::bind_downsample(module);
    bind_repeat(module);
    detail::bind_repeat_interleave(module);
    detail::bind_tilize(module);
    detail::bind_tilize_with_val_padding(module);
    detail::bind_tilize_with_zero_padding(module);
    detail::bind_transpose(module);
}

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
