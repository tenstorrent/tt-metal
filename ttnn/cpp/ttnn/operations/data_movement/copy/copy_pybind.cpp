// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "copy.hpp"

namespace ttnn::operations::data_movement::detail {

namespace py = pybind11;

std::string get_binary_doc_string(std::string op_name, std::string op_desc) {
    std::vector<std::string> arg_name = {"input_a", "input_b"};
    op_desc = fmt::format(fmt::runtime(op_desc), arg_name[0], arg_name[1]);

    std::string docstring = fmt::format(R"doc(
        {0}

        Both input tensors must be of equal shape.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "{2}", "First tensor to {1}", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "{3}", "Second tensor to {1}", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes")doc",
        op_desc, op_name, arg_name[0], arg_name[1]
    );
    return docstring;
}

std::string get_unary_doc_string(std::string op_name, std::string op_desc) {
    const std::string tensor_name = "input";
    op_desc = fmt::format(fmt::runtime(op_desc), tensor_name);
    std::string docstring = fmt::format(R"doc(
        {0}

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "{1}", "Tensor {2} is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes")doc",
        op_desc, tensor_name, op_name
    );

    return docstring;
}


void py_bind_copy(py::module& module) {
    auto doc =  get_binary_doc_string("copy", R"doc(  Copies the elements from ``{0}`` into ``{1}``. ``{1}`` is modified in place.)doc");

    bind_registered_operation(
        module,
        ttnn::copy,
        doc,
        ttnn::pybind_overload_t{
            [] (const decltype(ttnn::copy)& self,
                const ttnn::Tensor& input_a,
                const ttnn::Tensor& input_b,
                uint8_t queue_id) {
                    return self(queue_id, input_a, input_b);
                },
                py::arg("input_a").noconvert(),
                py::arg("input_b").noconvert(),
                py::kw_only(),
                py::arg("queue_id") = 0}
    );
}

void py_bind_clone(py::module& module) {
    auto doc = R"doc(clone(tensor: ttnn.Tensor, memory_config: MemoryConfig, dtype: DataType) -> ttnn.Tensor

    Clones the tensor by copying it with the given `memory config`. Also, converts the dataype to `dtype`.
    Note: clone does not change the layout of the tensor.
    Organizes the `ttnn.Tensor` :attr:`tensor` into either ROW_MAJOR_LAYOUT or TILE_LAYOUT.  When requesting ROW_MAJOR_LAYOUT
    the tensor will be returned unpadded in the last two dimensions.   When requesting TILE_LAYOUT the tensor will be automatically
    padded where the width and height become multiples of 32.
    In the case where the layout is the same, the operation simply pad or unpad the last two dimensions depending on layout requested.

    Args:
        * :attr:`tensor`: the ttnn.Tensor
        * :attr:`memory_config`: the `ttnn` memory config, DRAM_MEMORY_CONFIG or L1_MEMORY_CONFIG.
        * :attr:`dtype`: the `ttnn` data type.)doc";

    bind_registered_operation(
        module,
        ttnn::clone,
        doc,
        ttnn::pybind_overload_t{
            [] (const decltype(ttnn::clone)& self,
                const ttnn::Tensor& input_tensor,
                const std::optional<ttnn::MemoryConfig> &memory_config,
                const std::optional<const ttnn::DataType> dtype,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, memory_config, dtype);
                },
                py::arg("input_tensor").noconvert(),
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("dtype") = std::nullopt,
                py::arg("queue_id") = 0}
    );
}

void py_bind_assign(py::module& module) {
    auto doc = detail::get_unary_doc_string("assign", R"doc(  Returns a new tensor which is a new copy of input tensor ``{0}``.


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
        ttnn::pybind_overload_t{
            [] (const decltype(ttnn::assign)& self,
                const ttnn::Tensor& input,
                const ttnn::MemoryConfig memory_config,
                const std::optional<const ttnn::DataType> dtype,
                std::optional<ttnn::Tensor> &optional_output_tensor,
                uint8_t queue_id) {
                    return self(queue_id, input, memory_config, dtype, optional_output_tensor);
                },
                py::arg("input_tensor").noconvert(),
                py::kw_only(),
                py::arg("memory_config"),
                py::arg("dtype") = std::nullopt,
                py::arg("output_tensor") = std::nullopt,
                py::arg("queue_id") = 0},
        ttnn::pybind_overload_t{
            [] (const decltype(ttnn::assign)& self,
                const ttnn::Tensor& input_a,
                const ttnn::Tensor& input_b,
                uint8_t queue_id) {
                    return self(queue_id, input_a, input_b);
                },
                py::arg("input_a").noconvert(),
                py::arg("input_b").noconvert(),
                py::arg("queue_id") = 0}
    );
}

}  // namespace ttnn::operations::data_movement::detail
