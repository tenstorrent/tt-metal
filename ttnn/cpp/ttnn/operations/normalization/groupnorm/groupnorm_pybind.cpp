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

                GroupNorm traditionally operates by splitting the input tensor's channels into groups, and then computing the mean and variance of each group.
                This implementation is slightly different, in that it forms the groups using the tensor's last dimension.
                Concretely, the input tensor is expected to have a shape of [N, 1, H*W, C], where C is the dimension along which the groups are formed.

                TTNN provides utility functions to help prepare this op's inputs.
                    - When using sharded input tensors, :func:`ttnn.determine_expected_group_norm_sharded_config_and_grid_size` can provide the appropriate memory configuration and grid size.
                    - :func:`ttnn.create_group_norm_input_mask` creates the appropriate input mask for a given tensor dimension and group size.
                    - :func:`ttnn.create_group_norm_weight_bias_rm` converts the weight and bias tensors into appropriately padded and tiled inputs

                See the sharded example in this document for more details on how to properly prepare the op's inputs using these functions.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword args:
                num_groups (int): Number of groups to split the tensor's channels into.
                epsilon (float): Defaults to 1e-12.
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
                use_welford (bool, optional): Defaults to `False`. If `True`, the Welford's algorithm is used to compute the mean and variance.
                reciprocals (ttnn.Tensor, optional): Defaults to `None`. FP32 tensor containing pre-computed reciprocal values. Only valid when use_welford is True. Must be sharded to L1 memory in each core.

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
              - :attr:`input_tensor` is a 4D tensor of shape [N, 1, H*W, C] and is allocated on the device
              - For the :attr:`input_tensor`, N*H*W must be a multiple of the tile size (32) and C must divide evenly into :attr:`num_groups`.
              - For the :attr:`input_mask`, C must match the number of groups, H must match a tile's height, and W must be a multiple of a tile's width.
              - :attr:`gamma` and :attr:`beta` must be provided
              - :attr:`inplace` is not supported for TILE-layout inputs and requires input and output layouts to be identical.
              - When generating inputs (e.g. weight, bias) for block sharded tensors, the number of cores in a column should draw upon core.x rather than core.y.
              - When generating inputs (e.g. weight, bias) for height sharded tensors, the number of cores in a column should be 1 rather than core.y.
              - Width-sharding is not supported (use height or block sharding)

            Example (Sharded Input):
                .. code-block:: python

                     N, C, H, W = 1, 64, 32, 1
                    num_groups = 2

                    # Prepare random inputs
                    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
                    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
                    torch_bias = torch.rand((C,), dtype=torch.bfloat16)

                    # Generate random inputs and prepare reference output
                    torch_output_tensor = torch.nn.functional.group_norm(
                        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
                    )

                    # Permute the torch output to match the TTNN format, so they can be compared
                    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

                    #Prepare TTNN input
                    # Determine how to shard the input tensor
                    sharded_mem_config, grid_size = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
                        device = device,
                        num_channels = C,
                        num_groups = num_groups,
                        input_nhw = N * H * W,
                        is_height_sharded = True,
                        is_row_major = True
                    )

                    input_tensor = ttnn.from_torch(
                        torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C),
                        dtype=ttnn.DataType.BFLOAT16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=sharded_mem_config,
                    )

                    # Input mask - this is needed for each group to be able to select the correct elements of the input tensor
                    # In general, it will have dimensions of [1, num_groups, 32, 32*block_wt]

                    # In this example, C=64 and num_groups=2, so each group is 32 channels (i.e. one tile) wide
                    # As a result, the input_mask_tensor is a [1, 2, 32, 32] tensor where every value is 1

                    # If instead num_groups was 4, each group would be 16 channels (i.e. half a tile) wide
                    # As a result, the input_mask_tensor would be a [1, 4, 32, 32] tensor that selects either the first or second half of the tile
                    # e.g. The mask at [0][0][:][:] would be a 32x32 tensor where the left half is 1 and the right half is 0
                    # While [0][1][:][:] would be a 32x32 tensor where the left half is 0 and the right half is 1
                    input_mask_tensor = ttnn.create_group_norm_input_mask(
                        num_channels=C,
                        num_groups=num_groups,
                        num_cores_across_channel=1 # As explained in the Limitations, supply 1 for height sharded input tensors
                    )

                    input_mask_tensor = ttnn.from_torch(
                        input_mask_tensor,
                        dtype=ttnn.DataType.BFLOAT8_B,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

                    # Prepare gamma and beta for TTNN. Currently these are just 1D tensors of size [C], which isn't compatible with tile based processing
                    # First they will zero padded if needed (does not apply to this example)
                    # Then reshaped to be [1, 1, tiles_per_core_total, 32], which in this case will be [1, 1, 2, 32]

                    # As with the input mask, we supply a core count of 1 for height sharded input tensors
                    gamma = ttnn.create_group_norm_weight_bias_rm(input_tensor =torch_weight, num_channels = C, num_cores_x = 1)
                    beta = ttnn.create_group_norm_weight_bias_rm(input_tensor =torch_bias, num_channels = C, num_cores_x = 1)

                    gamma_t = ttnn.from_torch(
                        gamma,
                        dtype=ttnn.DataType.BFLOAT16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    beta_t = ttnn.from_torch(
                        beta,
                        dtype=ttnn.DataType.BFLOAT16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

                    # Compute the TTNN output and compare with the reference output
                    output_tensor = ttnn.group_norm(
                        input_tensor,
                        num_groups=num_groups,
                        input_mask=input_mask_tensor,
                        weight=gamma_t,
                        bias=beta_t,
                        memory_config=sharded_mem_config,
                        core_grid=grid_size,
                    )

                    output_tensor = ttnn.to_torch(output_tensor)
                    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)

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
            py::arg("reciprocals") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("core_grid") = std::nullopt,
            py::arg("inplace") = true,
            py::arg("output_layout") = std::nullopt,
            py::arg("num_out_blocks") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("negative_mask") = std::nullopt,
            py::arg("use_welford") = false});
}
void bind_normalization_group_norm(py::module& module) { bind_normalization_group_norm_operation(module); }

}  // namespace ttnn::operations::normalization::detail
