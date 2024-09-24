// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "loss_pybind.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loss.hpp"
#include "loss_types.hpp"
#include "ttnn/cpp/pybind11/export_enum.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace py = pybind11;

namespace ttnn::operations::loss {

namespace detail {

void bind_loss_type(py::module& m) {

    export_enum<LossReductionMode>(m, "LossReductionMode");
}

void bind_mse_loss_function(py::module& module) {
    auto doc = fmt::format(
        R"doc(
            Returns mean squared error loss function for `input_reference` and `input_prediction`

            Args:
                input_reference (ttnn.Tensor): the input tensor.
                input_prediction (ttnn.Tensor): the input tensor.


            Keyword Args:
                reduction (bool, optional): Loss Reduction Mode. Defaults to `None`.
                output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                ttnn.Tensor: the output tensor.

            Example:

                >>> input_reference = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
                >>> input_prediction = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
                >>> output = ttnn.mse_loss(input_reference, input_prediction, reduction)
        )doc",
        ttnn::mse_loss.base_name());

    using OperationType = decltype(ttnn::mse_loss);
    bind_registered_operation(
        module,
        ttnn::mse_loss,
        doc,
        ttnn::pybind_overload_t{
        [](const OperationType& self,
            const Tensor& ref,
            const Tensor& prediction,
            const LossReductionMode mode,
            const std::optional<MemoryConfig>& memory_config,
            std::optional<Tensor> optional_output_tensor,
            uint8_t queue_id) -> ttnn::Tensor {
                return self(queue_id, ref, prediction, mode, memory_config, optional_output_tensor);
            },
            py::arg("input_reference"),
            py::arg("input_prediction"),
            py::kw_only(),
            py::arg("reduction") = LossReductionMode::NONE,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0}
    );
}

void bind_mae_loss_function(py::module& module) {
    auto doc = fmt::format(
        R"doc(
            Returns mean absolute error loss function for `input_reference` and `input_prediction`

            Args:
                input_reference (ttnn.Tensor): the input tensor.
                input_prediction (ttnn.Tensor): the input tensor.


            Keyword Args:
                reduction (bool, optional): Loss Reduction Mode. Defaults to `None`.
                output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                ttnn.Tensor: the output tensor.

            Example:

                >>> input_reference = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
                >>> input_prediction = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
                >>> output = ttnn.l1_loss(input_reference, input_prediction, reduction)
        )doc",
        ttnn::l1_loss.base_name());

    using OperationType = decltype(ttnn::l1_loss);
    bind_registered_operation(
        module,
        ttnn::l1_loss,
        doc,
        ttnn::pybind_overload_t{
        [](const OperationType& self,
            const Tensor& ref,
            const Tensor& prediction,
            const LossReductionMode mode,
            const std::optional<MemoryConfig>& memory_config,
            std::optional<Tensor> optional_output_tensor,
            uint8_t queue_id) -> ttnn::Tensor {
                return self(queue_id, ref, prediction, mode, memory_config, optional_output_tensor);
            },
            py::arg("input_reference"),
            py::arg("input_prediction"),
            py::kw_only(),
            py::arg("reduction") = LossReductionMode::NONE,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0}
    );
}

} // detail


void py_bind_loss_functions(py::module& module) {
   detail::bind_loss_type(module);
   detail::bind_mse_loss_function(module);
   detail::bind_mae_loss_function(module);
}

} // ttnn::operations::loss
