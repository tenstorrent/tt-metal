// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_pybind.hpp"

#include <optional>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "minimal_matmul.hpp"
#include "ttnn-pybind/decorators.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::minimal_matmul::detail {

void py_bind_minimal_matmul(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::experimental::minimal_matmul,
        R"doc(
        TODO
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::experimental::minimal_matmul)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               const std::optional<ttnn::Tensor>& bias_tensor,
               const MinimalMatmulConfig& config,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
                return self(input_tensor, weight_tensor, bias_tensor, config, memory_config, compute_kernel_config);
            },
            py::kw_only(),
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("bias_tensor") = std::nullopt,
            py::arg("config"),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});

    auto py_minimal_matmul_config =
        py::class_<MinimalMatmulConfig>(
            module,
            "MinimalMatmulConfig",
            R"doc(
                            Configuration for the MinimalMatmul operation.
                            )doc")
            .def(py::init<>())
            .def(py::init<CoreCoord>(), py::kw_only(), py::arg("compute_with_storage_grid_size") = CoreCoord{1, 1});

    py_minimal_matmul_config.def_readwrite(
        "compute_with_storage_grid_size", &MinimalMatmulConfig::compute_with_storage_grid_size, "");

    py_minimal_matmul_config.def(
        "__repr__", [](const MinimalMatmulConfig& config) { return fmt::format("{}", config); });
}

}  // namespace ttnn::operations::experimental::minimal_matmul::detail
