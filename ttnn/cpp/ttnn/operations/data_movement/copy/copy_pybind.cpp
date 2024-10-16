// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "copy.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace {
std::string get_binary_doc_string(std::string_view op_name,
                                  std::string_view input_a,
                                  std::string_view input_b,
                                  fmt::format_string<std::string_view&, std::string_view&> op_desc) {
    return fmt::format(
        R"doc(
        {0}

        Both input tensors must be of equal shape.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "{2}", "First tensor to {1}", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "{3}", "Second tensor to {1}", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes")doc",
        fmt::format(op_desc, input_a, input_b),
        op_name,
        input_a,
        input_b);
}

std::string get_unary_doc_string(std::string_view op_name,
                                 std::string_view input,
                                 fmt::format_string<std::string_view&> op_desc) {
    return fmt::format(
        R"doc(
        {0}

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "{1}", "Tensor {2} is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes")doc",
        fmt::format(op_desc, input),
        input,
        op_name);
}
}  // namespace

namespace ttnn::operations::data_movement::detail {

namespace py = pybind11;

void py_bind_copy(py::module& module) {
    auto doc =
        get_binary_doc_string("copy",
                              "input_a",
                              "input_b",
                              R"doc(Copies the elements from ``{0}`` into ``{1}``. ``{1}`` is modified in place.)doc");

    bind_registered_operation(module,
                              ttnn::copy,
                              doc,
                              ttnn::pybind_overload_t{[](const decltype(ttnn::copy)& self,
                                                         const ttnn::Tensor& input_a,
                                                         const ttnn::Tensor& input_b,
                                                         uint8_t queue_id) {
                                                          return self(queue_id, input_a, input_b);
                                                      },
                                                      py::arg("input_a").noconvert(),
                                                      py::arg("input_b").noconvert(),
                                                      py::kw_only(),
                                                      py::arg("queue_id") = 0});
}

void py_bind_assign(py::module& module) {
    auto doc = get_unary_doc_string(
        "assign", "input", R"doc(  Returns a new tensor which is a new copy of input tensor ``{0}``.


    Alternatively, copies input tensor ``input_a`` to ``input_b`` if their
    shapes and memory layouts match, and returns input_b tensor.

    Input tensors can be of any data type.

    Output tensor will be of same data type as Input tensor.

    .. csv-table::
        :header: "Argument", "Description", "Data type", "Valid range", "Required"

        "input_a", "Tensor assign is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
        "input_b", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
        "queue_id", "command queue id", "uint8_t", "Default is 0", "No"

    )doc");

    bind_registered_operation(
        module,
        ttnn::assign,
        doc,
        ttnn::pybind_overload_t{[](const decltype(ttnn::assign)& self,
                                   const ttnn::Tensor& input,
                                   const ttnn::MemoryConfig memory_config,
                                   const std::optional<const ttnn::DataType> dtype,
                                   std::optional<ttnn::Tensor>& optional_output_tensor,
                                   uint8_t queue_id) {
                                    return self(queue_id, input, memory_config, dtype, optional_output_tensor);
                                },
                                py::arg("input_tensor").noconvert(),
                                py::kw_only(),
                                py::arg("memory_config"),
                                py::arg("dtype") = std::nullopt,
                                py::arg("output_tensor") = std::nullopt,
                                py::arg("queue_id") = 0},
        ttnn::pybind_overload_t{[](const decltype(ttnn::assign)& self,
                                   const ttnn::Tensor& input_a,
                                   const ttnn::Tensor& input_b,
                                   uint8_t queue_id) {
                                    return self(queue_id, input_a, input_b);
                                },
                                py::arg("input_a").noconvert(),
                                py::arg("input_b").noconvert(),
                                py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::data_movement::detail
