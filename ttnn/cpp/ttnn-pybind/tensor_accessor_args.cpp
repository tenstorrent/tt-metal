// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor_args.hpp"

#include <optional>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn-pybind/export_enum.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <hostdevcommon/tensor_accessor/arg_config.hpp>

namespace ttnn::tensor_accessor_args {

void py_module_types(py::module& module) {
    py::class_<tt::tt_metal::TensorAccessorArgs>(module, "TensorAccessorArgs");
    export_enum<tensor_accessor::ArgConfig>(module);
}

void py_module(py::module& module) {
    static_cast<py::class_<tt::tt_metal::TensorAccessorArgs>>(module.attr("TensorAccessorArgs"))
        .def(
            py::init<const tt::tt_metal::Buffer&, tensor_accessor::ArgsConfig>(),
            py::arg("buffer"),
            py::arg("args_config") = tensor_accessor::ArgConfig::None,
            R"doc(
                Initialize a TensorAccessorArgs with a buffer and an optional args config.
            )doc")
        .def(
            "append_to",
            py::overload_cast<std::vector<uint32_t>&, std::vector<uint32_t>&>(
                &tt::tt_metal::TensorAccessorArgs::append_to, py::const_),
            py::arg("compile_time_args"),
            py::arg("common_runtime_args"),
            R"doc(
                Append the tensor accessor args to the compile time and common runtime args.
            )doc")
        .def(
            "append_to",
            py::overload_cast<std::vector<uint32_t>&>(&tt::tt_metal::TensorAccessorArgs::append_to, py::const_),
            py::arg("compile_time_args"),
            R"doc(
                Append the tensor accessor args to the compile time args.
            )doc")
        .def(
            "get_compile_time_args",
            &tt::tt_metal::TensorAccessorArgs::get_compile_time_args,
            R"doc(
                Get the compile time args.
            )doc")
        .def(
            "get_common_runtime_args",
            &tt::tt_metal::TensorAccessorArgs::get_common_runtime_args,
            R"doc(
                Get the common runtime args.
            )doc");
}

}  // namespace ttnn::tensor_accessor_args
