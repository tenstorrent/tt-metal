// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "view.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::reshape::detail {

void bind_view(nb::module_& mod) {
    const auto* doc = R"doc(
        Note:

        - It is recommended to use ttnn.reshape if you are not sure which operation to use
        - If this is the functionality required for your application, it will be called by ttnn.reshape
        - The following conditions must be met for the function not to corrupt your data:
            * the last dimension must not change
            * In Layout::TILE the second last two dimensions must not change OR there is no padding on the second last dimension

        Args:
            * input_tensor: Input Tensor.
            * shape: Shape of tensor.

        Returns:
            ttnn.Tensor: the output tensor with the new shape.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((2, 1, 4), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.experimental.view(tensor, (2, 1, 1, 4))
        )doc";

    ttnn::bind_function<"view", "ttnn.experimental.">(
        mod,
        doc,
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(const ttnn::Tensor&, int32_t, int32_t, int32_t, int32_t)>(
                &ttnn::experimental::view),
            nb::arg("input_tensor"),
            nb::arg("N"),
            nb::arg("C"),
            nb::arg("H"),
            nb::arg("W")),
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(const ttnn::Tensor&, const ttsl::SmallVector<int32_t>&)>(
                &ttnn::experimental::view),
            nb::arg("input_tensor"),
            nb::arg("shape")));

    const auto* alias_doc = R"doc(
        Zero-copy TILE view of an interleaved device tensor's buffer with a new shape and dtype.

        The view shares the input's memory and keeps it alive. The bytes are reinterpreted, not converted, so
        the view's values are meaningful only for data written through the view.

        The intended input is a paged cache of shape [num_blocks, ...]. Interleaved page p lives on bank
        p % num_banks, so a block holds the same bytes in the input and in the view only when the block's
        page count is a multiple of the bank count. alias_view therefore requires:

        * the input to have rank 3 or more, TILE layout and interleaved memory;
        * the page count of one dim-0 slice of the input to be a multiple of the bank count;
        * the bytes one dim-0 slice holds on each bank to be a whole number of view pages;
        * the view to have rank 3 or more, and each of its dim-0 slices to lie inside one dim-0 slice of
          the input or to cover whole dim-0 slices of the input;
        * the view to fit inside the input's allocation on every bank.

        Then dim-0 slice i of the input and view pages [i * n, (i + 1) * n) cover the same bytes, where
        n = (pages of one input slice) * (input page size) / (view page size). Example: on 8 DRAM banks a
        bfloat16 [N, 1, 1024, 256] input (256 tiles per block) can be viewed as float32 [N, 8, 128, 128]
        (128 tiles per block). On 7 banks the same view raises.

        Args:
            * input_tensor: device tensor that owns the buffer.
            * shape: shape of the view.
            * dtype: dtype of the view.
        )doc";
    ttnn::bind_function<"alias_view", "ttnn.experimental.">(
        mod,
        alias_doc,
        &ttnn::experimental::alias_view,
        nb::arg("input_tensor"),
        nb::arg("shape"),
        nb::arg("dtype"));
}

}  // namespace ttnn::operations::experimental::reshape::detail
