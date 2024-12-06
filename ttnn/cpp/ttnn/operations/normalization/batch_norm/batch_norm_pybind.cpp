// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_pybind.hpp"

#include "batch_norm.hpp"

#include "pybind11/decorators.hpp"
namespace py = pybind11;
namespace ttnn::operations::normalization::detail {
void bind_batch_norm_operation(pybind11::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::batch_norm,
        R"doc(
            Applies Spatial Batch Normalization over each channel on :attr:`input_tensor`.Currently support is provided for inference mode only.


        Args:
            input_tensor (ttnn.Tensor): the input tensor.


        Keyword args:
            eps (float, optional): Epsilon value. Defaults to `1e-05`.
            running_mean (ttnn.Tensor, optional): the running_mean required for inference mode. Defaults to `None`.
            running_var (ttnn.Tensor, optional): the running_var required for inference mode. Defaults to `None`.
            weight (ttnn.Tensor, optional): the weight or gamma value. Defaults to `None`.
            bias (ttnn.Tensor, optional): the bias or beta value. Defaults to `None`.
            training (bool, optional): Selection between training mode and inference (evaluation) mode. Defaults to `False` (Inference mode).
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::kw_only(),
            py::arg("running_mean") = std::nullopt,
            py::arg("running_var") = std::nullopt,
            py::arg("training") = false,
            py::arg("eps") = 1e-05,
            py::arg("weight") = std::nullopt,
            py::arg("bias") = std::nullopt,
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt

        });
}
}  // namespace ttnn::operations::normalization::detail
