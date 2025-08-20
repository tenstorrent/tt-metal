// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "maxpool3d_pybind.hpp"

#include <optional>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "maxpool3d.hpp"
#include "ttnn-pybind/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::maxpool3d::detail {

void py_bind_maxpool3d(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::experimental::maxpool3d,
        R"doc(
        Performs 3D max pooling on the input tensor.

        Expects Input Tensor in [N, T, H, W, C] format in ROW_MAJOR layout.
        Output will be in [N, T_out, H_out, W_out, C] format in ROW_MAJOR layout.

        Args:
            input_tensor (ttnn.Tensor): Input tensor with shape [N, T, H, W, C] in ROW_MAJOR layout.
            kernel_size (Tuple[int, int, int]): Size of the 3D pooling kernel (T, H, W). Default: (2, 2, 2).
            stride (Tuple[int, int, int]): Stride for the 3D pooling operation (T, H, W). Default: (2, 2, 2).
            padding (Tuple[int, int, int]): Padding for the 3D pooling operation (T, H, W). Default: (0, 0, 0).
            padding_mode (str): Padding mode, either "zeros" or "replicate". Default: "zeros".
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration.
            queue_id: Queue ID for the operation.

        Returns:
            ttnn.Tensor: Output tensor after 3D max pooling with shape [N, T_out, H_out, W_out, C].

        Example:
            >>> input_tensor = ttnn.ones([1, 8, 16, 16, 32], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            >>> output = ttnn.experimental.maxpool3d(input_tensor, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            >>> print(output.shape)  # [1, 4, 8, 8, 32]
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::experimental::maxpool3d)& self,
               const ttnn::Tensor& input_tensor,
               const std::array<uint32_t, 3>& kernel_size,
               const std::array<uint32_t, 3>& stride,
               const std::array<uint32_t, 3>& padding,
               const std::string& padding_mode,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
               const QueueId& queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    kernel_size,
                    stride,
                    padding,
                    padding_mode,
                    memory_config,
                    compute_kernel_config);
            },
            py::kw_only(),
            py::arg("input_tensor"),
            py::arg("kernel_size") = std::array<uint32_t, 3>{2, 2, 2},
            py::arg("stride") = std::array<uint32_t, 3>{2, 2, 2},
            py::arg("padding") = std::array<uint32_t, 3>{0, 0, 0},
            py::arg("padding_mode") = "zeros",
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::experimental::maxpool3d::detail
