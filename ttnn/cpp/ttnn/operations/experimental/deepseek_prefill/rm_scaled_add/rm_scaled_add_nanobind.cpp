// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rm_scaled_add_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/rm_scaled_add/rm_scaled_add.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_rm_scaled_add_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Performs element-wise scaled addition: output = A + B * scale

            This is an experimental operation that treats row-major (RM) tensors as tile-formatted
            data for efficient FPU processing. The tensor shape should have elements divisible by 1024.

            Args:
                input_a: First input tensor (A) - bfloat16, row-major
                input_b: Second input tensor (B) - bfloat16, row-major, same shape as A
                scale: Scalar multiplier for B

            Returns:
                Tensor: Result of A + B * scale

            Example:
                >>> a = ttnn.from_torch(torch.randn(1, 7168), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                >>> b = ttnn.from_torch(torch.randn(1, 7168), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                >>> result = ttnn.experimental.rm_scaled_add(a, b, 0.5)
        )doc";

    using OperationType = decltype(ttnn::rm_scaled_add);
    bind_registered_operation(
        mod,
        ttnn::rm_scaled_add,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_a,
               const ttnn::Tensor& input_b,
               float scale) { return self(input_a, input_b, scale); },
            nb::arg("input_a").noconvert(),
            nb::arg("input_b").noconvert(),
            nb::arg("scale")});
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
