// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "moreh_clip_grad_norm.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {

void bind_moreh_clip_grad_norm_operation(nb::module_& mod) {
    const auto* const doc =
        R"doc(
            Clips gradient norm of an iterable of tensors.

            This operation computes the total norm of gradients across all input tensors and clips
            them to a maximum value. It is commonly used in training to prevent exploding gradients.
            The operation modifies the input tensors in-place.

            The total norm is computed as:

            .. math::
                \text{total\_norm} = \left(\sum_{i} |\text{grad}_i|^{\text{norm\_type}}\right)^{1/\text{norm\_type}}

            If total_norm > max_norm, each gradient is scaled by:

            .. math::
                \text{clip\_coef} = \frac{\text{max\_norm}}{\text{total\_norm} + 10^{-6}}

            Args:
                inputs (List[ttnn.Tensor]): Input tensors containing gradients to be clipped. All tensors must be on the same device.
                max_norm (float): Maximum norm value to clip gradients.
                norm_type (float, optional): Type of the norm (e.g., 2.0 for L2 norm). Defaults to 2.0.
                error_if_nonfinite (bool, optional): If True, throws error when total norm is non-finite (NaN or Inf). Defaults to False.

            Keyword Args:
                total_norm (ttnn.Tensor, optional): Pre-allocated output tensor for total norm. If not provided, a new tensor is created.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for intermediate tensors. If not provided, inherits from input tensors.
                compute_kernel_config (DeviceComputeKernelConfig, optional): Compute kernel configuration for the operation.

            Returns:
                ttnn.Tensor: Total norm of the gradients before clipping (scalar tensor with shape [1]).

            Note:
                * Input tensors are modified in-place with clipped gradients.
                * The operation is performed in three steps:
                    1. Compute sum of |grad|^norm_type for all gradients
                    2. Compute total_norm = (sum)^(1/norm_type)
                    3. Scale each gradient by min(max_norm / (total_norm + 1e-6), 1.0)
                * For very large numbers of input tensors, the operation is batched based on device core count.

            Example:
                >>> # Clip gradients with L2 norm
                >>> gradients = [ttnn.from_torch(torch.randn(32, 64), device=device) for _ in range(3)]
                >>> total_norm = ttnn.moreh_clip_grad_norm(gradients, max_norm=1.0, norm_type=2.0)
                >>> print(f"Total norm before clipping: {total_norm}")
        )doc";

    ttnn::bind_function<"moreh_clip_grad_norm">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const std::vector<Tensor>&,
                float,
                float,
                bool,
                const std::optional<const Tensor>&,
                const std::optional<MemoryConfig>&,
                const std::optional<DeviceComputeKernelConfig>&>(&ttnn::moreh_clip_grad_norm),
            nb::arg("inputs").noconvert(),
            nb::arg("max_norm").noconvert(),
            nb::arg("norm_type").noconvert() = 2.0f,
            nb::arg("error_if_nonfinite").noconvert() = false,
            nb::kw_only(),
            nb::arg("total_norm").noconvert() = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config").noconvert() = nb::none()));
}

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm
