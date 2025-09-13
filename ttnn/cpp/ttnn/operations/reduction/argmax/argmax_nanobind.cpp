// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"

namespace ttnn::operations::reduction::detail {
void bind_reduction_argmax_operation(nb::module_& mod) {
    auto doc =
        R"doc(

            Returns the indices of the maximum value of elements in the ``input`` tensor
            If no ``dim`` is provided, it will return the indices of maximum value of all elements in given ``input``
            If no ``keepdim`` is provided, it will default to `False`.

            Currently this op only support dimension-specific reduction on last dimension.

            Input tensor support bfloat16, float32, uint32, int32, uint16 data types and ROW_MAJOR layout.

            Output tensor will have UINT32 data type.

            Equivalent pytorch code:

            .. code-block:: python

                return torch.argmax(input_tensor, dim=dim, keepdim=keepdim)

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword args:
                dim (int, optional): dimension to reduce. Defaults to `None`.
                keepdim (bool, optional): whether to keep the reduced dimension. Defaults to `False`.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                List of ttnn.Tensor: the output tensor.

        )doc";

    using OperationType = decltype(ttnn::argmax);
    bind_registered_operation(
        mod,
        ttnn::argmax,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<int> dim,
               const bool keepdim,
               const std::optional<CoreRangeSet>& sub_core_grids,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor> optional_output_tensor,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    dim,
                    keepdim,
                    sub_core_grids,
                    use_multicore,
                    memory_config,
                    optional_output_tensor);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("dim") = nb::none(),
            nb::arg("keepdim") = false,
            nb::kw_only(),
            nb::arg("sub_core_grids") = nb::none(),
            nb::arg("use_multicore") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::reduction::detail
