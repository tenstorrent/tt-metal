// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "tt_dnn/op_library/backward/backward_ops.hpp"

namespace tt::tt_metal::detail{
    void TensorModuleBackwardOPs( py::module & m_tensor){

    m_tensor.def("conj_bw", py::overload_cast<const Tensor&, const Tensor&, const MemoryConfig&>(&conj_bw),
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for conjugate for complex tensor ``input`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("complex_recip_bw", py::overload_cast<const Tensor&, const Tensor&, const MemoryConfig&>(&complex_recip_bw),
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for reciprocal of complex tensor ``input`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("complex_abs_bw", py::overload_cast<const Tensor&, const Tensor&, const MemoryConfig&>(&complex_abs_bw),
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for abs of complex ``input`` tensor with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Tensor add is applied to", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("angle_bw", py::overload_cast<const Tensor&, const Tensor&, bool, const MemoryConfig&>(&angle_bw),
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("is_complextensor").noconvert() = true, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Performs backward operations for angle for the ``input`` with given ``grad``

                Input tensors must have BFLOAT16 data type.

                Output tensor will have BFLOAT16 data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Valid range", "Required"

                    "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "input", "Input Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                    "is_complextensor", "True(default) if input is complex tensor", "bool", "True/False", "No"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");

    m_tensor.def("imag_bw", py::overload_cast<const Tensor&, const Tensor&, const MemoryConfig&>(&imag_bw),
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for imaginary part of complex tensor ``input`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("real_bw", py::overload_cast<const Tensor&, const Tensor&, const MemoryConfig&>(&real_bw),
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for real part of complex tensor ``input`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input", "Input Tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("polar_bw", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&>(&polar_bw),
            py::arg("grad").noconvert(), py::arg("input_a").noconvert(), py::arg("input_b").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for polar ``input_a`` and  ``input_b`` with given ``grad``

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "input_a", "absolute value of the complex tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "angle of the complex tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("complex_div_bw", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&>(&complex_div_bw),
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for division of complex tensors``input`` and ``other`` with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "input", "First input tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "other", "Second input Tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("complex_mul_bw", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&>(&complex_mul_bw),
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for multiplication of complex tensors``input`` and ``other`` with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "input", "First input tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "other", "Second input Tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("complex_add_bw", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, float, const MemoryConfig&>(&complex_add_bw),
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("alpha") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for addition of  complex tensors``input`` and ``other`` with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "input", "First input tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "other", "Second input Tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "alpha", "Alpha value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def("complex_sub_bw", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, float, const MemoryConfig&>(&complex_sub_bw),
            py::arg("grad").noconvert(), py::arg("input").noconvert(), py::arg("other").noconvert(), py::arg("alpha") = 1.0f, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs backward operations for subtraction of  complex tensors``input`` and ``other`` with given ``grad``.

            Input tensors must have BFLOAT16 data type.

            Output tensors will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "grad", "Gradient tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "input", "First input tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "other", "Second input Tensor", "Tensor", "Tensor of complex shape [W, Z, Y, X]", "Yes"
                "alpha", "Alpha value", "float", "default to 1.0f", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    }
}
