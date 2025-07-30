// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor_args.hpp"

#include <optional>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn-pybind/export_enum.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <hostdevcommon/tensor_accessor/arg_config.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::tensor_accessor_args {

void py_module_types(py::module& module) {
    export_enum<tensor_accessor::ArgConfig>(module);
    py::class_<tt::tt_metal::TensorAccessorArgs>(module, "TensorAccessorArgs");
    py::class_<tensor_accessor::ArgsConfig>(module, "ArgsConfig");
}

void py_module(py::module& module) {
    static_cast<py::class_<tensor_accessor::ArgsConfig>>(module.attr("ArgsConfig"))
        .def(py::init<>())
        .def(py::init<tensor_accessor::ArgConfig>())
        .def(py::init<tensor_accessor::ArgConfig, tensor_accessor::ArgConfig>())
        .def("test", &tensor_accessor::ArgsConfig::test)
        .def("set", &tensor_accessor::ArgsConfig::set)
        .def("raw", &tensor_accessor::ArgsConfig::raw)
        .def(py::self | tensor_accessor::ArgConfig())
        .def(py::self | py::self)
        .def(py::self & tensor_accessor::ArgConfig());

    static_cast<py::class_<tt::tt_metal::TensorAccessorArgs>>(module.attr("TensorAccessorArgs"))
        .def(
            py::init([](const ttnn::Tensor& tensor, tensor_accessor::ArgsConfig args_config) {
                return tt::tt_metal::TensorAccessorArgs(*tensor.buffer(), args_config);
            }),
            py::arg("tensor"),
            py::arg("args_config") = tensor_accessor::ArgsConfig(tensor_accessor::ArgConfig::None),
            R"doc(
                Initialize a TensorAccessorArgs with a buffer and an optional args config.
            )doc")
        .def(
            "append_to",
            [](const tt::tt_metal::TensorAccessorArgs& self,
               std::vector<uint32_t>& compile_time_args,
               std::vector<uint32_t>& common_runtime_args) {
                self.append_to(compile_time_args, common_runtime_args);
                return py::make_tuple(compile_time_args, common_runtime_args);
            },
            py::arg("compile_time_args"),
            py::arg("common_runtime_args"),
            R"doc(
                Append the tensor accessor args to the compile time and common runtime args.
            )doc")
        .def(
            "append_to",
            [](const tt::tt_metal::TensorAccessorArgs& self, std::vector<uint32_t>& compile_time_args) {
                self.append_to(compile_time_args);
                return compile_time_args;
            },
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
