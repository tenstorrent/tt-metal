// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "untilize.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_untilize(py::module& module) {
    auto doc =
        R"doc(
            Changes data layout of input tensor to ROW_MAJOR.

            Input tensor must be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

            Output tensor will be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword Args:

                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
                use_pack_untilize (bool, optional): Whether to use pack untilize. Defaults to `True`.
                sub_core_grids (ttnn.CoreRangeSet, optional): Sub core grids. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                List of ttnn.Tensor: the output tensor.

        )doc";

    using OperationType = decltype(ttnn::untilize);
    ttnn::bind_registered_operation(
        module,
        ttnn::untilize,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               bool use_multicore,
               bool use_pack_untilize,
               const std::optional<CoreRangeSet>&& sub_core_grids,
               QueueId queue_id) {
                return self(queue_id, input_tensor, memory_config, use_multicore, use_pack_untilize, sub_core_grids);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("use_multicore") = true,
            py::arg("use_pack_untilize") = true,
            py::arg("sub_core_grids") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}
}  // namespace ttnn::operations::data_movement::detail
