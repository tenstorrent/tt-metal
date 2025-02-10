// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "cpp/pybind11/decorators.hpp"

#include "conv3d_pybind.hpp"
#include "conv3d.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations::conv {
namespace conv3d {

void py_bind_conv3d(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::conv3d,
        R"doc(
        Conv 3D
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::conv3d)& self,
               const ttnn::Tensor& input_tensor,
               //    const ttnn::Tensor& weight_tensor,
               uint32_t out_channels,
               std::array<uint32_t, 3> kernel_size,
               std::array<uint32_t, 3> stride,
               std::array<uint32_t, 3> padding,
               std::string padding_mode,
               uint32_t groups,
               //    std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const MemoryConfig>& memory_config,
               const uint8_t& queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    // weight_tensor,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    padding_mode,
                    groups,
                    // bias_tensor,
                    memory_config);
            },
            py::kw_only(),
            py::arg("input_tensor"),
            // py::arg("weight_tensor"),
            py::arg("out_channels"),
            py::arg("kernel_size"),
            py::arg("stride"),
            py::arg("padding"),
            py::arg("padding_mode"),
            py::arg("groups") = 1,
            // py::arg("bias_tensor") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace conv3d
}  // namespace operations::conv
}  // namespace ttnn
