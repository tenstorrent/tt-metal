// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "batch_norm.hpp"

#include "ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

namespace ttnn::operations::normalization::detail {
void bind_batch_norm_operation(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::batch_norm,
        R"doc(
            Applies Spatial Batch Normalization over each channel on :attr:`input_tensor`. Inputs must be must be tilized and interleaved.


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
            queue_id (int, optional): command queue id. Defaults to 0.


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


        Example:

            >>> input_tensor = ttnn.from_torch(torch.rand([2, 3, 4, 5], dtype=torch.bfloat16)), layout=ttnn.TILE_LAYOUT, device=device)
            >>> running_mean = ttnn.from_torch(torch.rand([1, 3, 1, 1], dtype=torch.bfloat16)), layout=ttnn.TILE_LAYOUT, device=device)
            >>> running_var = ttnn.from_torch(torch.rand([1, 3, 1, 1], dtype=torch.bfloat16)), layout=ttnn.TILE_LAYOUT, device=device)
            >>> weight = ttnn.from_torch(torch.rand([1, 3, 1, 1], dtype=torch.bfloat16)), layout=ttnn.TILE_LAYOUT, device=device)
            >>> bias = ttnn.from_torch(torch.rand([1, 3, 1, 1], dtype=torch.bfloat16)), layout=ttnn.TILE_LAYOUT, device=device)
            >>> eps = 1e-05
            >>> momentum = 0.1
            >>> output = ttnn.batch_norm(input_tensor, running_mean = running_mean, running_var = running_var, weight = weight, bias = bias, eps = eps, momentum = momentum, training = True)


        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::kw_only(),
            nb::arg("running_mean") = std::nullopt,
            nb::arg("running_var") = std::nullopt,
            nb::arg("training") = false,
            nb::arg("eps") = 1e-05,
            nb::arg("momentum") = 0.1,
            nb::arg("weight") = std::nullopt,
            nb::arg("bias") = std::nullopt,
            nb::arg("output") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId});
}
}  // namespace ttnn::operations::normalization::detail
