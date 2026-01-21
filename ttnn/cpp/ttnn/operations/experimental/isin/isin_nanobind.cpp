// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isin_nanobind.hpp"

#include "isin.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::experimental::isin::detail {
void bind_isin_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            This operator returns a uint32 tensor with the same shape, layout, and memory config as the elements tensor, where elements are filled with 0xFFFFFFFF (true) or 0x00000000 (false) based on their presence in test_elements.
            Parameters:
                * `elements` (Tensor): Tensor of integers to be checked for presence in test_elements. The output will have 0xFFFFFFFF (true) or 0x00000000 (false) at each position depending on whether the corresponding element is present in test_elements (and the invert flag).
                * `test_elements` (Tensor): Tensor containing the values to be checked against. If an element from `elements` is present in `test_elements`, the corresponding output value will be 0xFFFFFFFF (unless inverted).
            Keyword Arguments:
                * `invert` (bool): If True, inverts the output so that elements present in test_elements are marked with 0x00000000 and others with 0xFFFFFFFF.
            Notes:
                * `assume_unique` (bool): Currently has no effect, but is reserved for potential future optimizations.
                * The input tensors should be interleaved and in DRAM.
                * Both input tensors can be of any specification.

            Example:
                >>> device = ttnn.open_device(device_id=0)
                >>> elements = ttnn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
                >>> test_elements = ttnn.Tensor([2, 3, 5, 7], device=device)
                >>> output = ttnn.experimental.isin(elements, test_elements)
                >>> # output is [[False, True, True], [False, True, False], [True, False, False]], use the `invert=True` flag to invert this effect
        )doc";

    bind_registered_operation(
        mod,
        ttnn::experimental::isin,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("elements").noconvert(),
            nb::arg("test_elements").noconvert(),
            nb::kw_only(),
            nb::arg("assume_unique") = false,
            nb::arg("invert") = false,
            nb::arg("out") = nb::none()});
}

}  // namespace ttnn::operations::experimental::isin::detail
