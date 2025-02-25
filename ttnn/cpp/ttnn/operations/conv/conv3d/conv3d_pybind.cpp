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
               const ttnn::Tensor& weight_tensor,
               const std::optional<ttnn::Tensor>& bias_tensor,
               const Conv3dConfig& config,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
               const uint8_t& queue_id) {
                return self(
                    queue_id, input_tensor, weight_tensor, bias_tensor, config, memory_config, compute_kernel_config);
            },
            py::kw_only(),
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("bias_tensor") = std::nullopt,
            py::arg("config"),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("queue_id") = 0});

    auto py_conv3d_config = py::class_<Conv3dConfig>(module, "Conv3dConfig");
    py_conv3d_config.def(
        py::init<
            DataType,
            DataType,
            Layout,
            uint32_t,
            uint32_t,
            uint32_t,
            uint32_t,
            uint32_t,
            std::array<uint32_t, 3>,
            std::array<uint32_t, 3>,
            std::array<uint32_t, 3>,
            std::string,
            uint32_t,
            CoreCoord>(),
        py::kw_only(),
        py::arg("dtype") = DataType::BFLOAT16,
        py::arg("weights_dtype") = DataType::BFLOAT16,
        py::arg("output_layout") = Layout::ROW_MAJOR,
        py::arg("T_out_block") = 1,
        py::arg("W_out_block") = 1,
        py::arg("H_out_block") = 1,
        py::arg("C_out_block") = 0,
        py::arg("output_channels"),
        py::arg("kernel_size"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("padding_mode") = "zeros",
        py::arg("groups") = 1,
        py::arg("compute_with_storage_grid_size") = CoreCoord{1, 1});

    py_conv3d_config.def_readwrite("dtype", &Conv3dConfig::dtype);
    py_conv3d_config.def_readwrite("weights_dtype", &Conv3dConfig::weights_dtype);
    py_conv3d_config.def_readwrite("output_layout", &Conv3dConfig::output_layout);
    py_conv3d_config.def_readwrite("T_out_block", &Conv3dConfig::T_out_block);
    py_conv3d_config.def_readwrite("W_out_block", &Conv3dConfig::W_out_block);
    py_conv3d_config.def_readwrite("H_out_block", &Conv3dConfig::H_out_block);
    py_conv3d_config.def_readwrite("C_out_block", &Conv3dConfig::C_out_block);
    py_conv3d_config.def_readwrite("output_channels", &Conv3dConfig::output_channels);
    py_conv3d_config.def_readwrite("kernel_size", &Conv3dConfig::kernel_size);
    py_conv3d_config.def_readwrite("stride", &Conv3dConfig::stride);
    py_conv3d_config.def_readwrite("padding", &Conv3dConfig::padding);
    py_conv3d_config.def_readwrite("padding_mode", &Conv3dConfig::padding_mode);
    py_conv3d_config.def_readwrite("groups", &Conv3dConfig::groups);
    py_conv3d_config.def_readwrite("compute_with_storage_grid_size", &Conv3dConfig::compute_with_storage_grid_size);

    py_conv3d_config.def("__repr__", [](const Conv3dConfig& config) { return fmt::format("{}", config); });
}

}  // namespace conv3d
}  // namespace operations::conv
}  // namespace ttnn
