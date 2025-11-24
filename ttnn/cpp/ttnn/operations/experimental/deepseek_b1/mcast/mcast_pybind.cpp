// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mcast_pybind.hpp"
#include "mcast.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

namespace py = pybind11;

namespace ttnn::operations::experimental::deepseek_b1::mcast::detail {

void bind_mcast(py::module& module) {
    auto doc = fmt::format(
        R"doc(
        Mcasts data from a single-core sharded input tensor to a multi-core sharded output tensor.

        This operation takes an input tensor that is sharded on a single core and mcasts
        the data to an output tensor that is sharded across multiple cores. The output shard
        size must match the input shard size.

        Args:
            input_tensor (ttnn.Tensor): Input tensor sharded on a single core (must be row major layout).
            output_tensor (ttnn.Tensor): Output tensor sharded across multiple cores with the same shard size as input.
            noc (int, optional): The NOC to use for the mcast operation. Defaults to `None`.
        Returns:
            ttnn.Tensor: The output tensor with mcasted data.

        Example:
            >>> # Mcast from single core to multiple cores
            >>> output = ttnn.experimental.deepseek_b1.mcast(
            ...     input_tensor,
            ...     output_tensor,
            ...     noc=0
            ... )
        )doc");

    using OperationType = decltype(ttnn::experimental::deepseek_b1::mcast);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::deepseek_b1::mcast,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& output_tensor,
               uint32_t noc) { return self(input_tensor, output_tensor, noc); },
            py::arg("input_tensor"),
            py::arg("output_tensor"),
            py::arg("noc"),
        });
}

}  // namespace ttnn::operations::experimental::deepseek_b1::mcast::detail
