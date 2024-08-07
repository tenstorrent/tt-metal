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
#include "ttnn/operations/data_movement/slice/slice_pybind.hpp"
#include "ttnn/operations/data_movement/tilize/tilize_pybind.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding_pybind.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave_pybind.hpp"
#include "ttnn/operations/data_movement/transpose/transpose_pybind.hpp"
#include "ttnn/operations/data_movement/split/split_pybind.hpp"
#include "ttnn/operations/data_movement/untilize/untilize_pybind.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding_pybind.hpp"
#include "ttnn/operations/data_movement/untilize_with_halo_v2/untilize_with_halo_v2_pybind.hpp"
#include "ttnn/operations/data_movement/non_zero_indices/non_zero_indices_pybind.hpp"
#include "ttnn/operations/data_movement/fill_rm/fill_rm_pybind.hpp"


namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace data_movement {

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
    detail::bind_pad(module);
    detail::bind_slice(module);
    bind_repeat(module);
    detail::bind_repeat_interleave(module);
    detail::bind_tilize(module);
    detail::bind_tilize_with_val_padding(module);
    detail::bind_tilize_with_zero_padding(module);
    detail::bind_transpose(module);
    detail::bind_split(module);
    detail::bind_untilize(module);
    detail::bind_untilize_with_unpadding(module);
    detail::bind_untilize_with_halo_v2(module);
    bind_non_zero_indices(module);
    bind_fill_rm(module);
}

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
