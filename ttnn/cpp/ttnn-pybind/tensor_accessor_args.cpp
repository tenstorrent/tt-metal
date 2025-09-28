// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor_args.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::tensor_accessor_args {

void py_module_types(py::module& module) { py::class_<tt::tt_metal::TensorAccessorArgs>(module, "TensorAccessorArgs"); }

void py_module(py::module& module) {
    static_cast<py::class_<tt::tt_metal::TensorAccessorArgs>>(module.attr("TensorAccessorArgs"))
        .def(
            py::init([](const ttnn::Tensor& tensor) { return tt::tt_metal::TensorAccessorArgs(*tensor.buffer()); }),
            py::arg("tensor"),
            R"doc(
                Initialize a TensorAccessorArgs with a buffer and an optional args config.
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
