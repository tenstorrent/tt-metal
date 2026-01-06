// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor_args.hpp"

#include <nanobind/nanobind.h>

#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::tensor_accessor_args {

void py_module_types(nb::module_& mod) { nb::class_<tt::tt_metal::TensorAccessorArgs>(mod, "TensorAccessorArgs"); }

void py_module(nb::module_& mod) {
    static_cast<nb::class_<tt::tt_metal::TensorAccessorArgs>>(mod.attr("TensorAccessorArgs"))
        .def(
            "__init__",
            [](tt::tt_metal::TensorAccessorArgs* t, const ttnn::Tensor& tensor) {
                new (t) tt::tt_metal::TensorAccessorArgs(*tensor.buffer());
            },
            nb::arg("tensor"),
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
