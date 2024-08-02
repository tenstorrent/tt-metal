// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/composite/composite_ops.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/loss/loss_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/optimizer/optimizer_ops.hpp"
#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"


namespace tt::tt_metal::detail {


void TensorModuleCompositeOPs(py::module& m_tensor) {

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
