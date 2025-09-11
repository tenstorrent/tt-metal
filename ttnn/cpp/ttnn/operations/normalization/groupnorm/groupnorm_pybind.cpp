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
            Computes group_norm over :attr:`input_tensor`.
            See `Group Normalization <https://arxiv.org/abs/1803.08494>`_ for more details.

            .. math::
                \text{group_norm}(x, \gamma, \beta, \epsilon) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta

            Where:
                - :math:`\mu` and :math:`\sigma^2` are the mean and variance of the input tensor, respectively
                - :math:`\gamma` and :math:`\beta` are the learnable scale and shift parameters, respectively
                - :math:`\epsilon` is a small constant.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

          Keyword args:
              num_groups (int)
              epsilon (float): 1e-12.
              input_mask (ttnn.Tensor, optional): Defaults to `None`. When processing the inputs, the mask is used to only look at the elements of the current group.
              weight (ttnn.Tensor, optional): Defaults to `None`.
              bias (ttnn.Tensor, optional): Defaults to `None`.
              memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
              dtype (ttnn.DataType, optional): Defaults to `None`.
              core_grid (CoreGrid, optional): Defaults to `None`.
              inplace (bool, optional): Defaults to `True`.
              output_layout (ttnn.Layout, optional): Defaults to `None`.
              num_out_blocks (int, optional): Defaults to `None`.
              compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration for the op. Defaults to `None`.
              negative_mask (ttnn.Tensor, optional): Defaults to `None`. Can be used only in row-major sharded input/output tensors. Used to reduce the number of CB's used in the sharded version of the kernel by overlapping the CB's used for tilized input and output. (The kernel is in fact row major variant, but is internally tilizing RM into tilized inputs).


            Returns:
                ttnn.Tensor: the output tensor.

            Note:

                The supported input data types and layouts:

                .. list-table:: input_tensor
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - BFLOAT16
                        - TILE, ROW_MAJOR


                .. list-table:: weight (gamma) and bias (beta)
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - BFLOAT16
                        - ROW_MAJOR

                .. list-table:: input_mask
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - BFLOAT16, BFLOAT8_B
                        - TILE

                .. list-table:: output_tensor
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - BFLOAT16
                        - TILE, ROW_MAJOR

            Limitations:
              - Inputs are 4D tensors already allocated on device.
              - :attr:`input_tensor` is of shape [N, 1, H*W, C]
              - :attr:`gamma` and :attr:`beta` must be provided
              - For the :attr:`input_tensor`, N*H*W must be a multiple of the tile size (32) and C must be a multiple of :attr:`num_groups`.
              - For the :attr:`input_mask`, C must match the number of groups, H must match a tile's height, and W must be a multiple of a tile's width.
              - :attr:`inplace` is not supported for TILE-layout inputs and requires input and output layouts to be identical.
              - When generating inputs (e.g. weight, bias) for block sharded tensors, the number of cores in a column should draw upon core.x rather than core.y.
              - Width-sharding is not supported (use height or block sharding)

              Example:

                .. code-block:: python

                    tile_size = 32
                    N, C, H, W = 1, 480, 1, 64
                    grid_size = ttnn.CoreGrid(y=1, x=1)
                    num_out_blocks = 1

                    num_groups = 8 # This must be a multiple of grid_size.y (1 in this example)

                    input_tensor_row_major = ttnn.rand([N, 1, H*W, C], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                    input_tensor_tilized = ttnn.tilize_with_zero_padding(input_tensor_row_major, use_multicore=True)

                    # input mask
                    width_per_group = C // num_groups # C must be a multiple of num_groups
                    max_tiles_group_can_span = 1 + math.ceil((width_per_group-1) / tile_size)
                    input_mask_tensor = ttnn.zeros([1, num_groups, tile_size, max_tiles_group_can_span * tile_size], dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.TILE_LAYOUT, device=device)

                    # gamma/beta
                    values_per_chunk = C // grid_size.y # 480 / 1 = 480. Note that 480 is a multiple of 32, so no padding up to the next tile is needed.
                    values_per_chunk_per_tile = values_per_chunk // tile_size # 480 / 32 = 15

                    gamma_beta = ttnn.rand([1, 1, values_per_chunk_per_tile, 32], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

                    # groupnorm
                    output_tensor = ttnn.group_norm(
                        input_tensor_tilized,
                        num_groups=num_groups,
                        input_mask=input_mask_tensor,
                        weight=gamma_beta,
                        bias=gamma_beta,
                        output_layout=ttnn.TILE_LAYOUT,
                        core_grid=grid_size,
                        inplace=False,
                        num_out_blocks=num_out_blocks,
                    )

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
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("negative_mask") = std::nullopt});
}
void bind_normalization_group_norm(py::module& module) { bind_normalization_group_norm_operation(module); }

}  // namespace ttnn::operations::normalization::detail
