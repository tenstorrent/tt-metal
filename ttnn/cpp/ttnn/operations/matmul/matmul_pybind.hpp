// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace matmul {

using namespace tt::operations::primary;

void py_module(py::module& module) {
    py::class_<MatmulProgramConfig>(module, "MatmulProgramConfig")
        .def("__repr__", [](const MatmulProgramConfig& config) { return fmt::format("{}", config); });

    py::class_<MatmulMultiCoreReuseProgramConfig>(module, "MatmulMultiCoreReuseProgramConfig")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("in0_block_w").noconvert(),
            py::arg("out_subblock_h").noconvert(),
            py::arg("out_subblock_w").noconvert(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert())
        .def_readwrite("compute_with_storage_grid_size", &MatmulMultiCoreReuseProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseProgramConfig::in0_block_w)
        .def_readwrite("out_subblock_h", &MatmulMultiCoreReuseProgramConfig::out_subblock_h)
        .def_readwrite("out_subblock_w", &MatmulMultiCoreReuseProgramConfig::out_subblock_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseProgramConfig::per_core_N)
        .def("__repr__", [](const MatmulMultiCoreReuseProgramConfig& config) { return fmt::format("{}", config); });

    py::class_<MatmulMultiCoreReuseMultiCastProgramConfig>(module, "MatmulMultiCoreReuseMultiCastProgramConfig")
        .def(
            py::init<
                CoreCoord,
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                bool,
                std::optional<UnaryWithParam>,
                bool>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("in0_block_w").noconvert(),
            py::arg("out_subblock_h").noconvert(),
            py::arg("out_subblock_w").noconvert(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert(),
            py::arg("transpose_mcast").noconvert(),
            py::arg("fused_activation"),
            py::arg("fuse_batch").noconvert() = true)
        .def_readwrite("compute_with_storage_grid_size", &MatmulMultiCoreReuseMultiCastProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseMultiCastProgramConfig::in0_block_w)
        .def_readwrite("out_subblock_h", &MatmulMultiCoreReuseMultiCastProgramConfig::out_subblock_h)
        .def_readwrite("out_subblock_w", &MatmulMultiCoreReuseMultiCastProgramConfig::out_subblock_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseMultiCastProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseMultiCastProgramConfig::per_core_N)
        .def_readwrite("transpose_mcast", &MatmulMultiCoreReuseMultiCastProgramConfig::transpose_mcast)
        .def_readwrite("fused_activation", &MatmulMultiCoreReuseMultiCastProgramConfig::fused_activation)
        .def_readwrite("fuse_batch", &MatmulMultiCoreReuseMultiCastProgramConfig::fuse_batch)
        .def("__repr__", [](const MatmulMultiCoreReuseMultiCastProgramConfig& config) {
            return fmt::format("{}", config);
        });

    py::class_<MatmulMultiCoreReuseMultiCast1DProgramConfig>(module, "MatmulMultiCoreReuseMultiCast1DProgramConfig")
        .def(
            py::init<
                CoreCoord,
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                bool,
                std::optional<UnaryWithParam>,
                bool>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("in0_block_w").noconvert(),
            py::arg("out_subblock_h").noconvert(),
            py::arg("out_subblock_w").noconvert(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert(),
            py::arg("fuse_batch").noconvert(),
            py::arg("fused_activation"),
            py::arg("mcast_in0").noconvert())
        .def_readwrite("compute_with_storage_grid_size", &MatmulMultiCoreReuseMultiCast1DProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseMultiCast1DProgramConfig::in0_block_w)
        .def_readwrite("out_subblock_h", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_subblock_h)
        .def_readwrite("out_subblock_w", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_subblock_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseMultiCast1DProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseMultiCast1DProgramConfig::per_core_N)
        .def_readwrite("fuse_batch", &MatmulMultiCoreReuseMultiCast1DProgramConfig::fuse_batch)
        .def_readwrite("fused_activation", &MatmulMultiCoreReuseMultiCast1DProgramConfig::fused_activation)
        .def_readwrite("mcast_in0", &MatmulMultiCoreReuseMultiCast1DProgramConfig::mcast_in0)
        .def("__repr__", [](const MatmulMultiCoreReuseMultiCast1DProgramConfig& config) {
            return fmt::format("{}", config);
        });

    py::class_<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>(
        module, "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig")
        .def(
            py::init<
                std::size_t,
                std::size_t,
                std::size_t,
                std::optional<UnaryWithParam>>(),
            py::kw_only(),
            py::arg("in0_block_w").noconvert(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert(),
            py::arg("fused_activation"))
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::in0_block_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::per_core_N)
        .def_readwrite("fused_activation", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::fused_activation)
        .def("__repr__", [](const MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig& config) {
            return fmt::format("{}", config);
        });

    module.def(
        "matmul",
        [](const ttnn::Tensor& input_tensor_a,
           const ttnn::Tensor& input_tensor_b,
           const bool transpose_a = false,
           const bool transpose_b = false,
           const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
           const std::optional<const DataType> dtype = std::nullopt,
           const std::optional<const ttnn::MatmulProgramConfig> program_config = std::nullopt,
           const std::optional<const std::string>& activation = std::nullopt,
           const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
           const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt) -> ttnn::Tensor {
            return ttnn::operations::matmul::matmul(
                input_tensor_a,
                input_tensor_b,
                /*bias=*/std::nullopt,
                transpose_a,
                transpose_b,
                program_config,
                memory_config,
                dtype,
                activation,
                compute_kernel_config,
                core_grid,
                /*propagate_is_b_batched=*/true);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("transpose_a") = false,
        py::arg("transpose_b") = false,
        py::arg("memory_config") = DRAM_MEMORY_CONFIG,
        py::arg("dtype") = std::nullopt,
        py::arg("program_config") = std::nullopt,
        py::arg("activation") = std::nullopt,
        py::arg("compute_kernel_config") = std::nullopt,
        py::arg("core_grid") = std::nullopt);

    module.def(
        "linear",
        [](const ttnn::Tensor& input_tensor_a,
           const ttnn::Tensor& input_tensor_b,
           const std::optional<const ttnn::Tensor>& bias = std::nullopt,
           const bool transpose_a = false,
           const bool transpose_b = false,
           const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
           const std::optional<const DataType> dtype = std::nullopt,
           const std::optional<const ttnn::MatmulProgramConfig> program_config = std::nullopt,
           const std::optional<const std::string>& activation = std::nullopt,
           const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
           const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt) -> ttnn::Tensor {
            return ttnn::operations::matmul::matmul(
                input_tensor_a,
                input_tensor_b,
                bias,
                transpose_a,
                transpose_b,
                program_config,
                memory_config,
                dtype,
                activation,
                compute_kernel_config,
                core_grid);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("bias") = std::nullopt,
        py::arg("transpose_a") = false,
        py::arg("transpose_b") = false,
        py::arg("memory_config") = DRAM_MEMORY_CONFIG,
        py::arg("dtype") = std::nullopt,
        py::arg("program_config") = std::nullopt,
        py::arg("activation") = std::nullopt,
        py::arg("compute_kernel_config") = std::nullopt,
        py::arg("core_grid") = std::nullopt);
}

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn
