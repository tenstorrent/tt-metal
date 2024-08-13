// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
}
}  // namespace tt::tt_metal::detail
