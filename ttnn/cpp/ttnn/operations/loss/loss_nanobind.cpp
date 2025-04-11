// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "loss_nanobind.hpp"

#include <optional>

#include <fmt/format.h>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>

#include "loss.hpp"
#include "loss_types.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::loss {

namespace {

void bind_loss_type(nb::module_& mod) { export_enum<LossReductionMode>(mod, "LossReductionMode"); }

void bind_mse_loss_function(nb::module_& mod) {
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
        mod,
        ttnn::mse_loss,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const Tensor& ref,
               const Tensor& prediction,
               const LossReductionMode mode,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> optional_output_tensor) -> ttnn::Tensor {
                return self(ref, prediction, mode, memory_config, optional_output_tensor);
            },
            nb::arg("input_reference"),
            nb::arg("input_prediction"),
            nb::kw_only(),
            nb::arg("reduction") = LossReductionMode::NONE,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

void bind_mae_loss_function(nb::module_& mod) {
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
        mod,
        ttnn::l1_loss,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const Tensor& ref,
               const Tensor& prediction,
               const LossReductionMode mode,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> optional_output_tensor) -> ttnn::Tensor {
                return self(ref, prediction, mode, memory_config, optional_output_tensor);
            },
            nb::arg("input_reference"),
            nb::arg("input_prediction"),
            nb::kw_only(),
            nb::arg("reduction") = LossReductionMode::NONE,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

}  // namespace

void bind_loss_functions(nb::module_& mod) {
    bind_loss_type(mod);
    bind_mse_loss_function(mod);
    bind_mae_loss_function(mod);
}

}  // namespace ttnn::operations::loss
