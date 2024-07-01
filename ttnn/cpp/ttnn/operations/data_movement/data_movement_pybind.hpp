// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/data_movement.hpp"
#include "ttnn/operations/data_movement/pad/pad_pybind.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace data_movement {

void bind_permute(py::module& module) {
const auto doc = R"doc(
Permutes :attr:`input_tensor` using :attr:`order`.

Args:
    * :attr:`input_tensor`: the input tensor
    * :attr:`order`: the desired ordering of dimensions.

Example:

    >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
    >>> output = ttnn.permute(tensor, (0, 1, 3, 2))
    >>> print(output.shape)
    [1, 1, 32, 64]

    )doc";

    module.def(
        "permute",
        &permute,
        py::arg("input_tensor"),
        py::arg("order"),
        doc);
}

void bind_concat(py::module& module) {
    const auto doc = R"doc(
Concats :attr:`tensors` in the given :attr:`dim`.

Args:
    * :attr:`tensors`: the tensors to be concatenated.
    * :attr:`dim`: the concatenating dimension.

Keyword Args:
    * :attr:`memory_config`: the memory configuration to use for the operation

Example:

    >>> tensor = ttnn.concat(ttnn.from_torch(torch.zeros((1, 1, 64, 32), ttnn.from_torch(torch.zeros((1, 1, 64, 32), dim=3)), device)

    >>> tensor1 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> tensor2 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> output = ttnn.concat([tensor1, tensor2], dim=4)
    >>> print(output.shape)
    [1, 1, 32, 64]

    )doc";
    module.def(
        "concat",
        &concat,
        py::arg("input_tensor"),
        py::arg("dim") = 0,
        py::kw_only(),
        py::arg("memory_config") = std::nullopt,
        doc);
}

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
            py::arg("input_tensor"),
            py::arg("scale_factor"),
            py::arg("memory_config") = std::nullopt});
}



void bind_repeat_interleave(py::module& module) {
    auto doc = R"doc(
repeat_interleave(input_tensor: ttnn.Tensor, repeats : int, dim: int = 0) -> ttnn.Tensor

Repeats elements of a :attr:`tensor` in the given :attr:`dim`.

Args:
    * :attr:`input_tensor`: the input_tensor to apply the repeate interleave operation.
    * :attr:`repeats`: The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
    * :attr:`dim`: the dimension to expand with the repetitions.

Example:

torch_input_tensor =
    torch_result = torch.repeat_interleave(torch_input_tensor, repeats, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.repeat_interleave(input_tensor, repeats, dim=dim)
    >>> a = ttnn.from_torch(torch.rand(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    >>> b = ttnn.repeat_interleave(a, 2, dim=0)
    >>> print(a.shape, b.shape)
    ttnn.Shape([1, 1, 32, 32]) ttnn.Shape([2, 1, 32, 32])
        )doc";

    ttnn::bind_registered_operation(
    module,
    ttnn::repeat_interleave,
    doc,
    ttnn::pybind_arguments_t{
        py::arg("input_tensor"),
        py::arg("repeats"),
        py::arg("dim"),
        py::kw_only(),
        py::arg("memory_config") = std::nullopt});
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
        py::arg("input_tensor"),
        py::arg("shape"),
        py::kw_only(),
        py::arg("memory_config") = std::nullopt});
}



void py_module(py::module& module) {
    bind_permute(module);
    bind_concat(module);
    bind_upsample(module);
    detail::bind_pad(module);
    bind_repeat(module);
    bind_repeat_interleave(module);
}

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
