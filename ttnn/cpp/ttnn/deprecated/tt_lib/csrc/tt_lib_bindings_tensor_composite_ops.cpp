// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/complex/complex_ops.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/composite/composite_ops.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/loss/loss_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/optimizer/optimizer_ops.hpp"
#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"


namespace tt::tt_metal::detail {


void TensorModuleCompositeOPs(py::module& m_tensor) {


    m_tensor.def(
        "outer",
        &outer,
        py::arg("input_a").noconvert(),
        py::arg("input_b").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
            Perform a non-batched outer product multiplication ``arg0 x arg1`` with two tensors.

            Both input tensors must have BFLOAT16 data type but shape [1,1,N,1] and [1,1,1,M] respectively
            or reshapeable with only one major dimension while other 3 being squeezable dimensions.

            Output tensor will have BFLOAT16 data type but of shape [1,1,N,M].

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_a", "First tensor to multiply", "Tensor", "Tensor of shape [1, 1, N, 1]", "Yes"
                "input_b", "Second tensor to multiply", "Tensor", "Tensor of shape [1, 1, 1, M]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        // *** composite unary ops ***
    detail::bind_unary_op_with_param(
        m_tensor,
        "polyval",
        &polyval,
        py::arg("coeffs"),
        R"doc(Returns tensor with the polyval of all of elements of the input tensor ``{0}`` with coefficients ``{1}``.)doc",
        R"doc("coefficients value with highest degree first", "List of float", "List size > 0")doc");

    m_tensor.def(
        "lerp",
        py::overload_cast<const Tensor&, const Tensor&, float, const MemoryConfig&>(&lerp),
        py::arg("input").noconvert(),
        py::arg("end").noconvert(),
        py::arg("weight"),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
            Applies the linear interpolation of two tensors ``input`` and ``end`` based on a
            scalar ``weight`` and returns the resulting out tensor.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor lerp is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "end", "End value", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "weight", "Weight value", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def(
        "lerp",
        py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&>(&lerp),
        py::arg("input").noconvert(),
        py::arg("end").noconvert(),
        py::arg("weight").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
            Applies the linear interpolation of two tensors ``input`` and ``end`` based on a
            tensor ``weight`` and returns the resulting out tensor.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor lerp is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "end", "End value", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "weight", "Weight value", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");



#if 0
        m_tensor.def("bitwise_complement", &bitwise_complement, R"doc(
            Returns tensor with the bitwise complement of elements of the input tensor ``arg0``.

            Input tensor must have UINT32 data type.

            Output tensor will have UINT32 data type.

            +----------+---------------------------+-----------+------------------------------+----------+
            | Argument | Description               | Data type | Valid range                  | Required |
            +==========+===========================+===========+==============================+==========+
            | arg0     | Tensor bitwise complement |           |                              |          |
            |          | '~' is applied to         | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
            +----------+---------------------------+-----------+------------------------------+----------+
        )doc");


        m_tensor.def("logical_not", &logical_not, R"doc(
            Returns tensor with the logical notof elements of the input tensor ``arg0``.

            Input tensor must have UINT32 data type.

            Output tensor will have UINT32 data type.

            +----------+---------------------------+-----------+------------------------------+----------+
            | Argument | Description               | Data type | Valid range                  | Required |
            +==========+===========================+===========+==============================+==========+
            | arg0     | Tensor logical not        |           |                              |          |
            |          | '!' is applied to         | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
            +----------+---------------------------+-----------+------------------------------+----------+
        )doc");
#endif

#if 0
        m_tensor.def("mean", &mean, R"doc(
            Returns tensor with the mean of elements of the input tensor ``arg0``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            +----------+---------------------------+-----------+------------------------------+----------+
            | Argument | Description               | Data type | Valid range                  | Required |
            +==========+===========================+===========+==============================+==========+
            | arg0     | Tensor mean is computed   | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
            +----------+---------------------------+-----------+------------------------------+----------+
        )doc");

        m_tensor.def("std", &tt::tt_metal::std, R"doc(
            Returns tensor with the std of elements of the input tensor ``arg0``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            +----------+---------------------------+-----------+------------------------------+----------+
            | Argument | Description               | Data type | Valid range                  | Required |
            +==========+===========================+===========+==============================+==========+
            | arg0     | Tensor std is computed on | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
            +----------+---------------------------+-----------+------------------------------+----------+
        )doc");

        m_tensor.def("normalize", &normalize, R"doc(
            Returns tensor with the normalization of elements of the input tensor ``arg0``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            +----------+---------------------------+-----------+------------------------------+----------+
            | Argument | Description               | Data type | Valid range                  | Required |
            +==========+===========================+===========+==============================+==========+
            | arg0     | Tensor std normalized     | Tensor    | Tensor of shape [W, Z, Y, X] | Yes      |
            +----------+---------------------------+-----------+------------------------------+----------+
        )doc");
#endif
        
        m_tensor.def("unary_rdiv_trunc", py::overload_cast<float, const Tensor&, const MemoryConfig&>(&unary_rdiv_trunc),
            py::arg("value").noconvert(), py::arg("input").noconvert(),  py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Performs the element-wise division of a scalar ``value`` by a tensor ``input`` and rounds the result using trunc mode. Support provided only for Wormhole_B0.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "value", "Numerator value", "float", "", "Yes"
                "input", "Denominator Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");


        m_tensor.def("rfloor_div", py::overload_cast<float, const Tensor&, const MemoryConfig&>(&rfloor_div),
            py::arg("value").noconvert(), py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,R"doc(
            Performs the element-wise floor division of a scalar ``value`` by a tensor ``input``. Support provided only for Wormhole_B0.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "value", "Numerator value", "float", "", "Yes"
                "input", "Denominator Tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("mac", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&>(&mac),
            py::arg("input").noconvert(), py::arg("tensor1").noconvert(), py::arg("tensor2").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns tensor with the multiply and accumulation of all of elements of the input tensors ``input, tensor1, tensor2``.
            Output is ``input x tensor1 + tensor2`` elementwise operator.
            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor mac is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tensor1", "Tensor to be multiplied", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "tensor2", "Tensor to be added", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def(
        "mac",
        py::overload_cast<const Tensor&, float, float, const MemoryConfig&>(&mac),
        py::arg("input").noconvert(),
        py::arg("float1"),
        py::arg("float2"),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
            Returns tensor with the multiply and accumulation of all of elements of the input tensor ``input11 with``float1, float2``.
            Output is ``tensor1 x float1 + float2`` elementwise operator.
            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Tensor mac is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "float1", "Value to be multiplied", "float", "", "Yes"
                "float2", "Value to be added", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

    m_tensor.def(
        "lamb_optimizer",
        &lamb_optimizer,
        py::arg("data").noconvert(),
        py::arg("grad").noconvert(),
        py::arg("exp_avg").noconvert(),
        py::arg("exp_avg_sq").noconvert(),
        py::arg("beta1"),
        py::arg("beta2"),
        py::arg("step_size"),
        py::arg("eps"),
        py::arg("weight_decay"),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
            Returns tensor with the threshold activation on elements of the input tensors ``arg0`` at threshold ``threshold``,
            and value ``value``.

            Input tensor must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "data", "Tensor data is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "grad", "Tensor grad is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "exp_avg", "Tensor exp_avg is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "exp_avg_sq", "exp_avg_sq threshold is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "beta1", "Value to beta1 at", "float", "", "Yes"
                "beta2", "Value to beta2 with", "float", "", "Yes"
                "step_size", "Value to beta1 at", "float", "", "Yes"
                "eps", "Value to beta2 with", "float", "", "Yes"
                "weight_decay", "Value to beta1 at", "float", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        // *** complex operations ***
        detail::bind_unary_op(m_tensor, "angle", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::angle), R"doc(Returns elementwise angle of complex tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "real", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::real), R"doc(Returns real portion of complex tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "imag", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::imag), R"doc(Returns imag portion of complex tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "is_real", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::is_real), R"doc(Returns true if complex tensor ``{0}``  is real.)doc");
        detail::bind_unary_op(m_tensor, "is_imag", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::is_imag), R"doc(Returns true if complex tensor ``{0}``  is imaginary.)doc");
        detail::bind_unary_op(m_tensor, "complex_abs", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::complex_abs), R"doc(Returns elementwise abs value of complex tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "conj", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::conj), R"doc(Returns elementwise complex conjugate of tensor ``{0}``.)doc");
        detail::bind_unary_op(m_tensor, "complex_recip", py::overload_cast<const Tensor&,const MemoryConfig&>(&tt::tt_metal::complex_recip), R"doc(Returns elementwise reciprocal of complex tensor ``{0}``.)doc");

        m_tensor.def("complex_mul", py::overload_cast<const Tensor&,const Tensor&,const MemoryConfig&>(&tt::tt_metal::complex_mul),
            py::arg("input_a"), py::arg("input_b"),
            py::arg("output_mem_config").noconvert() = std::nullopt,R"doc(Perform an eltwise-binary multiplication ``input_a * input_b`` on two complex tensors.)doc");

        m_tensor.def("complex_div", py::overload_cast<const Tensor&,const Tensor&,const MemoryConfig&>(&tt::tt_metal::complex_div),
            py::arg("input_a"), py::arg("input_b"),
            py::arg("output_mem_config").noconvert() = std::nullopt,R"doc(Perform an eltwise-binary divide ``input_a / input_b`` on two complex tensors.)doc");

        m_tensor.def("complex_add", py::overload_cast<const Tensor&,const Tensor&,const MemoryConfig&>(&tt::tt_metal::complex_add),
            py::arg("input_a"), py::arg("input_b"),
            py::arg("output_mem_config").noconvert() = std::nullopt,R"doc(Perform an eltwise-binary addition ``input_a + input_b`` on two complex tensors.)doc");
        m_tensor.def("complex_sub", py::overload_cast<const Tensor&,const Tensor&,const MemoryConfig&>(&tt::tt_metal::complex_sub),
            py::arg("input_a"), py::arg("input_b"),
            py::arg("output_mem_config").noconvert() = std::nullopt,R"doc(Perform an eltwise-binary subtraction ``input_a - input_b`` on two complex tensors.)doc");

        m_tensor.def("polar", py::overload_cast<const Tensor&,const Tensor&, const MemoryConfig&>(&tt::tt_metal::polar),
	    py::arg("input_a"), py::arg("input_b"),
            py::arg("output_mem_config").noconvert() = std::nullopt,R"doc(Perform an polar to Cartesian transformation of the input_a (r), input_b(theta) into x + i*y generating a type-2 complex tensor.)doc");

        detail::bind_binary_op<false, true, false, false>(m_tensor, "scatter", &tt::tt_metal::scatter, R"doc(Performs scatter operation on elements of the input tensors ``{0}`` and ``{1}``,specifically to copy channel data.)doc");


    // loss functions
    m_tensor.def(
        "mseloss",
        py::overload_cast<const Tensor&, const Tensor&, const LossReductionMode, const MemoryConfig&>(
            tt::tt_metal::mseloss),
        py::arg("input_reference"),
        py::arg("input_prediction"),
        py::arg("reduce_mode"),
        py::arg("output_mem_config").noconvert() = std::nullopt,
        R"doc(Returns mean squared error loss function for ``{0}`` and ``{1}``.)doc");

    m_tensor.def(
        "maeloss",
        py::overload_cast<const Tensor&, const Tensor&, const LossReductionMode, const MemoryConfig&>(
            tt::tt_metal::maeloss),
        py::arg("input_reference"),
        py::arg("input_prediction"),
        py::arg("reduce_mode"),
        py::arg("output_mem_config").noconvert() = std::nullopt,
        R"doc(Returns mean absolute error loss function for ``{0}`` and ``{1}``.)doc");
}
}  // namespace tt::tt_metal::detail
