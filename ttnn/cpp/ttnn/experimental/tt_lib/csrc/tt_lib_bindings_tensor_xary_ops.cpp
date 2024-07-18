// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"

namespace tt::tt_metal::detail {
    void TensorModuleXaryOPs( py::module & m_tensor){
        // *** eltwise unary ***

        detail::bind_unary_op_with_param(
            m_tensor, "unary_chain", &unary_chain,
            py::arg("unary_chain"),
            R"doc(Returns tensor with the unary op chain applied to all of elements of the input tensor ``{0}``.)doc",
            R"doc("Unary op chain", "Vector<FusibleActivation>", "At least 1 activation")doc"
        );

}
}  // namespace tt::tt_metal::detail
