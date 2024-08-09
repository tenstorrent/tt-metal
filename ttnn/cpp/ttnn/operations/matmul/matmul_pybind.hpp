// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tt_metal/common/core_coord.h"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace matmul {

using namespace tt::operations::primary;
using ttnn::operations::unary::UnaryWithParam;

template <typename T>
auto tt_pybind_class(py::module m, auto name, auto desc) {
    return py::class_<T>(m, name, desc)
        .def("to_json", [](const T& self) -> std::string { return tt::stl::json::to_json(self).dump(); })
        .def(
            "from_json",
            [](const std::string& json_string) -> T { return tt::stl::json::from_json<T>(nlohmann::json::parse(json_string)); })
        .def("__repr__", [](const T& self) { return fmt::format("{}", self); });
}

void py_module(py::module& module) {
    auto matmul_program_config = tt_pybind_class<MatmulProgramConfig>(module, "MatmulProgramConfig", R"doc(
        Class defining matmul program config
    )doc");

    auto matmul_multi_core_reuse_program_config = tt_pybind_class<MatmulMultiCoreReuseProgramConfig>(module, "MatmulMultiCoreReuseProgramConfig", R"doc(
        Class defining matmul multi core reuse program config
    )doc");

    matmul_multi_core_reuse_program_config.def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("in0_block_w").noconvert(),
            py::arg("out_subblock_h").noconvert(),
            py::arg("out_subblock_w").noconvert(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert())
        .def_readwrite(
            "compute_with_storage_grid_size", &MatmulMultiCoreReuseProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseProgramConfig::in0_block_w)
        .def_readwrite("out_subblock_h", &MatmulMultiCoreReuseProgramConfig::out_subblock_h)
        .def_readwrite("out_subblock_w", &MatmulMultiCoreReuseProgramConfig::out_subblock_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseProgramConfig::per_core_N);

    auto matmul_multi_core_reuse_multicast_program_config = tt_pybind_class<MatmulMultiCoreReuseMultiCastProgramConfig>(module, "MatmulMultiCoreReuseMultiCastProgramConfig", R"doc(
        Class defining matmul multi core reuse multi cast program config
    )doc");

    matmul_multi_core_reuse_multicast_program_config.def(
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
        .def_readwrite(
            "compute_with_storage_grid_size",
            &MatmulMultiCoreReuseMultiCastProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseMultiCastProgramConfig::in0_block_w)
        .def_readwrite("out_subblock_h", &MatmulMultiCoreReuseMultiCastProgramConfig::out_subblock_h)
        .def_readwrite("out_subblock_w", &MatmulMultiCoreReuseMultiCastProgramConfig::out_subblock_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseMultiCastProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseMultiCastProgramConfig::per_core_N)
        .def_readwrite("transpose_mcast", &MatmulMultiCoreReuseMultiCastProgramConfig::transpose_mcast)
        .def_readwrite("fused_activation", &MatmulMultiCoreReuseMultiCastProgramConfig::fused_activation)
        .def_readwrite("fuse_batch", &MatmulMultiCoreReuseMultiCastProgramConfig::fuse_batch);

    auto matmul_multi_core_reuse_multicast_1d_program_config = tt_pybind_class<MatmulMultiCoreReuseMultiCast1DProgramConfig>(module, "MatmulMultiCoreReuseMultiCast1DProgramConfig", R"doc(
        Class defining matmul multi core reuse multi cast 1D program config
    )doc");

    matmul_multi_core_reuse_multicast_1d_program_config.def(
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
        .def_readwrite(
            "compute_with_storage_grid_size",
            &MatmulMultiCoreReuseMultiCast1DProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseMultiCast1DProgramConfig::in0_block_w)
        .def_readwrite("out_subblock_h", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_subblock_h)
        .def_readwrite("out_subblock_w", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_subblock_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseMultiCast1DProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseMultiCast1DProgramConfig::per_core_N)
        .def_readwrite("fuse_batch", &MatmulMultiCoreReuseMultiCast1DProgramConfig::fuse_batch)
        .def_readwrite("fused_activation", &MatmulMultiCoreReuseMultiCast1DProgramConfig::fused_activation)
        .def_readwrite("mcast_in0", &MatmulMultiCoreReuseMultiCast1DProgramConfig::mcast_in0);

    auto matmul_multi_core_reuse_multicast_dram_sharded_program_config = tt_pybind_class<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>(module, "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig", R"doc(
        Class defining matmul multi core reuse multi cast DRAM sharded program config
    )doc");

    matmul_multi_core_reuse_multicast_dram_sharded_program_config.def(
            py::init<std::size_t, std::size_t, std::size_t, std::optional<UnaryWithParam>>(),
            py::kw_only(),
            py::arg("in0_block_w").noconvert(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert(),
            py::arg("fused_activation"))
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::in0_block_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::per_core_N)
        .def_readwrite("fused_activation", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::fused_activation);

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
            std::optional<CoreCoord> user_core_coord;
            if (core_grid.has_value()) {
                user_core_coord = CoreCoord(core_grid->x, core_grid->y);
            }
            bool user_run_batched = ttnn::operations::matmul::detail::is_input_batched(input_tensor_b.get_shape());
            return ttnn::operations::matmul::matmul(
                input_tensor_a,
                input_tensor_b,
                /*bias=*/std::nullopt,
                Matmul{
                    program_config,
                    /*bcast_batch=*/std::nullopt,
                    memory_config,
                    dtype,
                    compute_kernel_config,
                    /*untilize_out=*/false,
                    user_core_coord,
                    get_fused_activation(activation),
                    user_run_batched,
                    transpose_a,
                    transpose_b});
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
            std::optional<CoreCoord> user_core_coord;
            if (core_grid.has_value()) {
                user_core_coord = CoreCoord(core_grid->x, core_grid->y);
            }
            bool b_is_batched = ttnn::operations::matmul::detail::is_input_batched(input_tensor_b.get_shape());
            TT_FATAL(
                !(b_is_batched && bias.has_value()),
                "Batched input not supported when bias exists (linear operation).");

            return ttnn::operations::matmul::matmul(
                input_tensor_a,
                input_tensor_b,
                bias,
                tt::operations::primary::Matmul{
                    program_config,
                    /*bcast_batch=*/std::nullopt,
                    memory_config,
                    dtype,
                    compute_kernel_config,
                    /*untilize_out=*/false,
                    user_core_coord,
                    get_fused_activation(activation),
                    /*user_run_batched=*/false,
                    transpose_a,
                    transpose_b});
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
