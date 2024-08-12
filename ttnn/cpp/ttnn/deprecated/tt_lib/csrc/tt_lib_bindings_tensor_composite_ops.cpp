// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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


}
}  // namespace tt::tt_metal::detail
