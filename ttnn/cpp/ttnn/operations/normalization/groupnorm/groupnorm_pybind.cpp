// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "groupnorm.hpp"

namespace ttnn::operations::normalization::detail {
namespace py = pybind11;

void bind_normalization_group_norm_operation(pybind11::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::group_norm,
        R"doc(
            Computes group_norm over :attr:`input_tensor`, as described in `Group Normalization <https://arxiv.org/abs/1803.08494>`_.

            .. math::
                \text{group_norm}(x, \gamma, \beta, \epsilon) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta

            where :math:`\mu` and :math:`\sigma^2` are the mean and variance of the input tensor, respectively.
            :math:`\gamma` and :math:`\beta` are the learnable scale and shift parameters, respectively.
            :math:`\epsilon` is a small constant.

            Notes:
            - input_tensor must be of shape [N, 1, H*W, C] to match convolution-friendly memory layout.
           - Data type and formats:
              - Operates on BF16 tensors located on-device.
              - Output dtype must match input dtype; leave dtype=None unless you already match the input.
            - Optional weight (gamma) and bias (beta):
              - If used, provide in ROW_MAJOR layout in BF16, and also on-device.
              - If both are provided, they should share the same layout and dtype. Shapes must broadcast across the channel dimension.
            - input_mask (optional):
              - Use TILE layout with the group dimension equal to num_groups.
              - The mask tile height must match the tile height; the width must be a whole number of tiles per group.
            - Tiling and shape alignment: choose N, H, W so that N*H*W fits exactly into an integer number of tiles.
            - In-place usage:
              - Not supported for TILE-layout inputs.
              - Requires input and output layouts to be identical.
            - When generating inputs (e.g. mask, weight, and bias) for block sharded tensors, the number of cores in a column should draw upon core.x rather than core.y.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            num_groups (int): Number of groups to split the channels into. Must evenly divide C.
            epsilon (float): Defaults to `1e-12`.
            input_mask (ttnn.Tensor, optional): Defaults to `None`.
            weight (ttnn.Tensor, optional): Defaults to `None`.
            bias (ttnn.Tensor, optional): Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            dtype (ttnn.DataType, optional): Defaults to `None`.
            core_grid (CoreGrid, optional): Must be specified; automatic grid selection is not supported. Defaults to `None`.
            inplace (bool, optional): Defaults to `True`.
            output_layout (ttnn.Layout, optional): Defaults to `None`.
            num_out_blocks (int, optional): Applies to unsharded runs to split work across cores; ignored for sharded inputs. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration for the op. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.


        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("num_groups"),
            py::arg("epsilon") = 1e-12,
            py::arg("input_mask") = std::nullopt,
            py::arg("weight") = std::nullopt,
            py::arg("bias") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("core_grid") = std::nullopt,
            py::arg("inplace") = true,
            py::arg("output_layout") = std::nullopt,
            py::arg("num_out_blocks") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
void bind_normalization_group_norm(py::module& module) { bind_normalization_group_norm_operation(module); }

}  // namespace ttnn::operations::normalization::detail
