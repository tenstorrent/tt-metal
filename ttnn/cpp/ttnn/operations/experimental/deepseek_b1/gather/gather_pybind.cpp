// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_pybind.hpp"
#include "gather.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

namespace py = pybind11;

namespace ttnn::operations::experimental::deepseek_b1::gather::detail {

void bind_gather(py::module& module) {
    auto doc = fmt::format(
        R"doc(
        Gathers data from a single-core sharded input tensor to a multi-core sharded output tensor.

        This operation takes an input tensor that is sharded on a single core and gathers
        the data to an output tensor that is sharded across multiple cores. The output shard
        size must match the input shard size.

        Args:
            input_tensor (ttnn.Tensor): Input tensor sharded on a single core (must be row major layout).
            output_tensor (ttnn.Tensor): Output tensor sharded across multiple cores with the same shard size as input.
            noc (int, optional): The NOC to use for the gather operation. Defaults to `None`.
        Returns:
            ttnn.Tensor: The output tensor with gathered data.

        Example:
            >>> # Gather from single core to multiple cores
            >>> output = ttnn.experimental.deepseek_b1.gather(
            ...     input_tensor,
            ...     output_tensor,
            ...     noc=0
            ... )
        )doc");

    using OperationType = decltype(ttnn::experimental::deepseek_b1::gather);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::deepseek_b1::gather,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& output_tensor,
               std::optional<uint32_t> noc) { return self(input_tensor, output_tensor, noc); },
            py::arg("input_tensor"),
            py::arg("output_tensor"),
            py::arg("noc") = std::nullopt,
        });
}

}  // namespace ttnn::operations::experimental::deepseek_b1::gather::detail
