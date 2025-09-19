// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_pybind.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "batch_norm.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::normalization::detail {

void bind_batch_norm_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::batch_norm,
        R"doc(
        Applies batch norm over each channel on :attr:`input_tensor`.
        See `Spatial Batch Normalization <https://arxiv.org/abs/1502.03167>`_ for more details.

        .. math::

            \text{batch_norm}(x, \gamma, \beta, \epsilon) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta

        Where:
            - :math:`\mu` and :math:`\sigma^2` are the mean and variance of the input tensor, respectively
            - :math:`\gamma` and :math:`\beta` are the learnable scale and shift parameters, respectively
            - :math:`\epsilon` is a small constant.

        Args:
            input_tensor (ttnn.Tensor): the input tensor of shape `[N, C, H, W]`.

        Keyword args:
            eps (float, optional): Epsilon value. Defaults to `1e-05`.
            momentum (float, optional): Momentum value. Defaults to `0.1`.
            running_mean (ttnn.Tensor, optional): the running_mean of shape `[1, C, 1, 1]`, required in inference mode. When in training mode, this tensor is optional and the updated running mean value is stored in-place based on the inputs provided. Defaults to `None`.
            running_var (ttnn.Tensor, optional): the running_var of shape `[1, C, 1, 1]`, required in inference mode. When in training mode, this tensor is optional and the updated running variance value is stored in-place based on the inputs provided. Defaults to `None`.
            weight (ttnn.Tensor, optional): the weight or gamma value of shape `[1, C, 1, 1]`. Defaults to `None`.
            bias (ttnn.Tensor, optional): the bias or beta value of shape `[1, C, 1, 1]`. Defaults to `None`.
            training (bool, optional): Selection between training mode and inference (evaluation) mode. Defaults to `False` (Inference mode).
            output (ttnn.Tensor, optional): Preallocated output tensor to store batch norm result of shape `[N, C, H, W]`. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): device compute kernel configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - BFLOAT16, FLOAT32
                 - TILE
                 - 4

            These apply for all the tensor inputs to this operation, including the optional :attr:`output` tensor.

        Limitations:
            - All input tensors must be tilized, interleaved, rank 4, and on-device.

        Example:
            .. code-block:: python

                N, C, H, W = 2, 3, 4, 5

                input_tensor = ttnn.rand([N, C, H, W], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
                running_mean = ttnn.rand([1, C, 1, 1], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
                running_var = ttnn.rand([1, C, 1, 1], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
                weight = ttnn.rand([1, C, 1, 1], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
                bias = ttnn.from_torch(torch.rand([1, C, 1, 1], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

                output = ttnn.batch_norm(
                    input_tensor,
                    running_mean = running_mean,
                    running_var = running_var,
                    weight = weight,
                    bias = bias,
                    eps = 1e-05,
                    momentum = 0.1,
                    training = True
                )

        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::kw_only(),
            py::arg("running_mean") = std::nullopt,
            py::arg("running_var") = std::nullopt,
            py::arg("training") = false,
            py::arg("eps") = 1e-05,
            py::arg("momentum") = 0.1,
            py::arg("weight") = std::nullopt,
            py::arg("bias") = std::nullopt,
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::normalization::detail
