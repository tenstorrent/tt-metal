// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "groupnorm.hpp"
#include "groupnorm_input_mask.hpp"

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

                The output will be BFLOAT16, and both the layout and the memory configuration will match the :attr:`input_tensor`.

            Memory Support:
              - Interleaved: DRAM and L1
              - Sharded (L1): Height and Block sharded

            Limitations:
              - :attr:`input_tensor` is a 4D tensor of shape [N, 1, H*W, C] and is allocated on the device
              - For the :attr:`input_tensor`, N*H*W must be a multiple of the tile size (32) and C must divide evenly into :attr:`num_groups`.
              - For the :attr:`input_mask`, C must match the number of groups, H must match a tile's height, and W must be a multiple of a tile's width.
              - :attr:`gamma` and :attr:`beta` must be provided
              - :attr:`inplace` is not supported for TILE-layout inputs and requires input and output layouts to be identical.
              - When generating inputs (e.g. weight, bias) for block sharded tensors, the number of cores in a column should draw upon core.x rather than core.y.
              - When generating inputs (e.g. weight, bias) for height sharded tensors, the number of cores in a column should be 1 rather than core.y.
              - Width-sharding is not supported
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
    module.def(
        "create_group_norm_input_mask",
        [](int64_t num_channel, int64_t num_groups, int64_t num_cores_across_channel, DataType data_type = DataType::BFLOAT16) {
            return create_group_norm_input_mask(num_channel, num_groups, num_cores_across_channel, data_type);
        },
        py::arg("num_channel"),
        py::arg("num_groups"),
        py::arg("num_cores_across_channel"),
        py::arg("data_type") = DataType::BFLOAT16,
        R"doc(
            C++ implementation of create_group_norm_input_mask.
            Returns a ttnn.Tensor of shape [1, num_groups, 32, 32*block_wt], dtype=ttnn.DataType.BFLOAT16.
        )doc"
    );
    module.def(
        "create_group_norm_input_negative_mask",
        [](int64_t num_channel, int64_t num_groups, int64_t num_cores_across_channel, DataType data_type = DataType::BFLOAT16) {
            return create_group_norm_input_negative_mask(num_channel, num_groups, num_cores_across_channel, data_type);
        },
        py::arg("num_channel"),
        py::arg("num_groups"),
        py::arg("num_cores_across_channel"),
        py::arg("data_type") = DataType::BFLOAT16,
        R"doc(
            C++ implementation of create_group_norm_input_negative_mask.
            Returns a ttnn.Tensor of shape [1, num_groups, 32, 32*block_wt], dtype=ttnn.DataType.BFLOAT16.
        )doc"
    );
}
void bind_normalization_group_norm(py::module& module) { bind_normalization_group_norm_operation(module); }

}  // namespace ttnn::operations::normalization::detail
