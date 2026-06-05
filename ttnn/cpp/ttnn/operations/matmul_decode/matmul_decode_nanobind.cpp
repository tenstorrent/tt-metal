// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/matmul_decode/matmul_decode.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::matmul_decode {

void bind_matmul_decode_operation(nb::module_& mod) {
    ttnn::bind_function<"matmul_decode">(
        mod,
        R"doc(matmul_decode(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, partial_width_sharded: bool = False, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

        Returns the matrix product of two tensors.

        Args:
            input_tensor_a (ttnn.Tensor): the first tensor to be multiplied.
            input_tensor_b (ttnn.Tensor): the second tensor to be multiplied.

        Keyword Args:
            partial_width_sharded (bool, optional): force the partial width-sharded program
                factory, where B is sharded along both K and N and the K-partials are reduced
                across cores. Defaults to False (factory chosen automatically).
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to None.

        Returns:
            ttnn.Tensor: the output tensor.
        )doc",
        &ttnn::matmul_decode,
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::kw_only(),
        nb::arg("partial_width_sharded") = false,
        nb::arg("dtype") = nb::none());
}

}  // namespace ttnn::operations::matmul_decode
