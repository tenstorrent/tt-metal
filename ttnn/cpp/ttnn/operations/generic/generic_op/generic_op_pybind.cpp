// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/stl_bind.h>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "generic_op.hpp"
#include "generic_op_types.hpp"
#include "generic_op_pybind.hpp"

#include "pybind11/export_enum.hpp"
#include "pybind11/decorators.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::generic {

tt::tt_metal::ComputeConfig convert_to_compute_config(const ttnn::WormholeComputeKernelConfig& wormholeConfig) {
    tt::tt_metal::ComputeConfig compute_config;
    compute_config.math_fidelity = wormholeConfig.math_fidelity;
    compute_config.fp32_dest_acc_en = wormholeConfig.fp32_dest_acc_en;
    compute_config.dst_full_sync_en = wormholeConfig.dst_full_sync_en;
    compute_config.math_approx_mode = wormholeConfig.math_approx_mode;
    compute_config.compile_args = wormholeConfig.compile_args;
    compute_config.defines = wormholeConfig.defines;
    return compute_config;
}

void py_module_types(py::module& module) {
    py::class_<circular_buffer_attributes_t>(module, "CircularBufferAttributes")
        .def(py::init<>())
        .def(
            py::init<CoreRangeSet, uint32_t, uint32_t, ttnn::DataType>(),
            py::arg("core_spec"),
            py::arg("total_size"),
            py::arg("page_size"),
            py::arg("dtype"))
        .def_readwrite("core_spec", &circular_buffer_attributes_t::core_spec)
        .def_readwrite("total_size", &circular_buffer_attributes_t::total_size)
        .def_readwrite("page_size", &circular_buffer_attributes_t::page_size)
        .def_readwrite("dtype", &circular_buffer_attributes_t::data_format);

    py::class_<data_movement_attributes_t>(module, "DataMovementAttributes")
        .def(py::init<>())
        .def(
            py::init([](CoreRangeSet core_spec,
                        std::string kernel_path,
                        std::vector<uint32_t> config,
                        std::unordered_map<CoreCoord, std::vector<uint32_t>> runtime_args_per_core,
                        bool is_reader) {
                // Convert std::vector<uint32_t> to ReaderDataMovementConfig
                if (is_reader) {
                    return data_movement_attributes_t{
                        core_spec, kernel_path, tt::tt_metal::ReaderDataMovementConfig(config), runtime_args_per_core};
                } else {
                    return data_movement_attributes_t{
                        core_spec, kernel_path, tt::tt_metal::WriterDataMovementConfig(config), runtime_args_per_core};
                }
            }),
            py::arg("core_spec"),
            py::arg("kernel_path"),
            py::arg("config"),
            py::arg("runtime_args_per_core"),
            py::arg("is_reader") = true)
        .def_readwrite("core_spec", &data_movement_attributes_t::core_spec)
        .def_readwrite("kernel_path", &data_movement_attributes_t::kernel_path)
        .def_readwrite("config", &data_movement_attributes_t::config)
        .def_readwrite("runtime_args_per_core", &data_movement_attributes_t::runtime_args_per_core);

    py::class_<compute_attributes_t>(module, "ComputeAttributes")
        .def(py::init<>())
        .def(
            py::init([](CoreRangeSet core_spec,
                        std::string kernel_path,
                        ttnn::WormholeComputeKernelConfig config,
                        std::unordered_map<CoreCoord, std::vector<uint32_t>> runtime_args_per_core) {
                // Convert ttnn::WormholeComputeKernelConfig to tt::tt_metal::ComputeConfig
                return compute_attributes_t{
                    core_spec, kernel_path, convert_to_compute_config(config), runtime_args_per_core};
            }),
            py::arg("core_spec"),
            py::arg("kernel_path"),
            py::arg("config"),
            py::arg("runtime_args_per_core") = std::unordered_map<CoreCoord, std::vector<uint32_t>>())
        // .def(py::init<
        //     CoreRangeSet,
        //     std::string,
        //     tt::tt_metal::ComputeConfig,
        //     std::unordered_map<CoreCoord, std::vector<uint32_t>>>(),
        //     py::arg("core_spec"),
        //     py::arg("kernel_path"),
        //     py::arg("config"),
        //     py::arg("runtime_args_per_core"))
        .def_readwrite("core_spec", &compute_attributes_t::core_spec)
        .def_readwrite("kernel_path", &compute_attributes_t::kernel_path)
        .def_readwrite("config", &compute_attributes_t::config)
        .def_readwrite("runtime_args_per_core", &compute_attributes_t::runtime_args_per_core);

    py::class_<program_attributes_t>(module, "ProgramAttributes")
        .def(py::init<>())
        .def(
            py::init<
                std::unordered_map<tt::CBIndex, circular_buffer_attributes_t>,
                std::vector<data_movement_attributes_t>,
                std::vector<compute_attributes_t>>(),
            py::arg("circular_buffer_attributes"),
            py::arg("data_movement_attributes"),
            py::arg("compute_attributes"))
        .def_readwrite("circular_buffer_attributes", &program_attributes_t::circular_buffer_attributes)
        .def_readwrite("data_movement_attributes", &program_attributes_t::data_movement_attributes)
        .def_readwrite("compute_attributes", &program_attributes_t::compute_attributes);

    export_enum<tt::CBIndex>(module, "CBIndex");
    py::implicitly_convertible<py::int_, tt::CBIndex>();
}

void py_module(py::module& module) {
    std::string doc =
        R"doc(
        Generates a tensor to draw binary random numbers (0 or 1) from a Bernoulli distribution.

        Args:
            io_tensors (ttnn.Tensor): List of input tensors and output tensor. Output tensor must be the last element.
            program_attributes

        Returns:
            ttnn.Tensor: handle to the output tensor.

        Example:
            >>> input = ttnn.to_device(ttnn.from_torch(torch.empty(3, 3).uniform_(0, 1), dtype=torch.bfloat16)), device=device)
            >>> program_attributes =
            >>> output = ttnn.bernoulli(input)

        )doc";

    bind_registered_operation(
        module, ttnn::generic_op, doc, ttnn::pybind_arguments_t{py::arg("io_tensors"), py::arg("program_attributes")});
}

}  // namespace ttnn::operations::generic
