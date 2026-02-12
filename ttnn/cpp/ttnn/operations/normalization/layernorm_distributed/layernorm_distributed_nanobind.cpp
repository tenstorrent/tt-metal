// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_distributed_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "layernorm_pre_all_gather.hpp"
#include "layernorm_post_all_gather.hpp"

namespace ttnn::operations::normalization::detail {

void bind_normalization_layernorm_pre_all_gather_operation(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::layer_norm_pre_all_gather,
        R"doc(
              This operation is used in conjunction with :func:`ttnn.layer_norm_post_all_gather` to compute layer norm on a distributed setup, where layer norm is defined as:

              .. math::
                \text{layer_norm}(x, \gamma, \beta, \epsilon) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta

              Where:
                  - :math:`\mu` is the mean of the input tensor. This is computed over the last dimension of the input tensor (W).
                  - :math:`\sigma^2` is the variance of the input tensor. This is computed over the last dimension of the input tensor (W) and is biased.
                  - :math:`\gamma` and :math:`\beta` are the learnable scale and shift parameters, respectively
                  - :math:`\epsilon` is a small constant

              See `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ for more details.

              This operation computes :math:`\sum_{}^{}x` and :math:`\sum_{}^{}x^2` over the last dimension.
              Its output should be combined across devices with :func:`ttnn.all_gather`, then followed by :func:`ttnn.layer_norm_post_all_gather` to compute layer norm.

              Args:
                input_tensor (ttnn.Tensor): the input tensor.

              Keyword args:
                dtype (ttnn.DataType, optional): the data type of the output tensor. Defaults to BFLOAT16.
                residual_input_tensor (ttnn.Tensor, optional): the residual input tensor. Defaults to None.
                compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): the compute kernel configuration. Defaults to None.
                program_config (ttnn.ProgramConfig, optional): the program configuration. Defaults to None.
                memory_config (ttnn.MemoryConfig, optional): the memory configuration. Defaults to None.

              Returns:
                ttnn.Tensor: the output tensor.

              Note:
                Supported data types and layouts by tensor:

                .. list-table:: input_tensor
                  :header-rows: 1

                  * - dtype
                    - layout
                  * - BFLOAT16, FLOAT32, BFLOAT8_B
                    - TILE

                .. list-table:: residual_input_tensor
                  :header-rows: 1

                  * - dtype
                    - layout
                  * - BFLOAT16, FLOAT32, BFLOAT8_B
                    - TILE

                Output stats tensor will be in TILE layout and have dtype of BFLOAT16.

              Limitations:
                - Input tensors must be on-device and rank 4.
                - Unsharded runs: :attr:`input_tensor` must be interleaved.
                - Sharded runs: inputs cannot be height-sharded, padded height must equal TILE_HEIGHT (32).
                - When using :attr:`residual_input_tensor` with sharding, it must match the :attr:`input_tensor` padded shape and sharding.
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("dtype") = nb::cast(DataType::BFLOAT16),
            nb::arg("residual_input_tensor") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("memory_config") = nb::none()});
}

void bind_normalization_layernorm_post_all_gather_operation(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::layer_norm_post_all_gather,
        R"doc(
                This operation is used in conjunction with :func:`ttnn.layer_norm_pre_all_gather` to compute layer norm on a distributed setup, where layer norm is defined as:

                .. math::
                  \text{layer_norm}(x, \gamma, \beta, \epsilon) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta

                Where:
                    - :math:`\mu` is the mean of the input tensor. This is computed over the last dimension of the input tensor (W).
                    - :math:`\sigma^2` is the variance of the input tensor. This is computed over the last dimension of the input tensor (W) and is biased.
                    - :math:`\gamma` and :math:`\beta` are the learnable scale and shift parameters, respectively
                    - :math:`\epsilon` is a small constant

                See `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ for more details.

                Performs the second part of a distributed layernorm operation, using the gathered statistics to compute the mean and variance, and finally normalizing the input.
                The input :attr:`stats` tensor should be computed by first using :func:`ttnn.layer_norm_pre_all_gather` and then using :func:`ttnn.all_gather` to gather the statistics across all devices.

                Args:
                  input_tensor (ttnn.Tensor): the input tensor.
                  stats (ttnn.Tensor): the stats tensor.

                Keyword args:
                  epsilon (float, optional): the epsilon value. Defaults to 1e-12.
                  weight (ttnn.Tensor, optional): the weight tensor. Defaults to None.
                  bias (ttnn.Tensor, optional): the bias tensor. Defaults to None.
                  memory_config (ttnn.MemoryConfig, optional): the memory configuration. Defaults to None.
                  compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): the compute kernel configuration. Defaults to None.
                  program_config (ttnn.ProgramConfig, optional): the program configuration. Defaults to None.
                  dtype (ttnn.DataType, optional): the data type of the output tensor. Defaults to None.

                Returns:
                  ttnn.Tensor: the output tensor.

                Note:
                  Supported data types and layouts:

                  .. list-table:: input_tensor
                    :header-rows: 1

                    * - dtype
                      - layout
                    * - BFLOAT16, BFLOAT8_B
                      - TILE

                  .. list-table:: stats
                    :header-rows: 1

                    * - dtype
                      - layout
                    * - BFLOAT16
                      - TILE

                  .. list-table:: weight (gamma) and bias (beta)
                    :header-rows: 1

                    * - dtype
                      - layout
                    * - BFLOAT16
                      - ROW_MAJOR

                  Output tensor will be in TILE layout and have the same dtype as the :attr:`input_tensor`

                Limitations:
                  - Input tensors must be on-device and rank 4.
                  - The last padded dim of :attr:`stats` must be a multiple of TILE_WIDTH.
                  - The first three padded dims of :attr:`stats` must match :attr:`input_tensor`.
                  - If :attr:`weight` (gamma) is provided, :attr:`bias` (beta) must also be provided with matching layouts with their last padded dim matching TILE_WIDTH.
                  - Sharded runs: inputs cannot be height-sharded, padded height must equal TILE_HEIGHT (32), and :attr:`stats` must be sharded with `num_cores=1` and expected tile columns per device.
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("stats"),
            nb::kw_only(),
            nb::arg("epsilon") = 1e-12,
            nb::arg("weight") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("dtype") = nb::none()});
}

void bind_normalization_layernorm_distributed(nb::module_& mod) {
    bind_normalization_layernorm_pre_all_gather_operation(mod);
    bind_normalization_layernorm_post_all_gather_operation(mod);
}

}  // namespace ttnn::operations::normalization::detail
