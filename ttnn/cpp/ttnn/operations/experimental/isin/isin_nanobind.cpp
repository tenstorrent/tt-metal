// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "isin_nanobind.hpp"

#include "isin.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::isin::detail {
void bind_isin_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            This operator returns a uint32 tensor with the same shape, layout, and memory config as the elements tensor, where elements are filled with 0xFFFFFFFF (true) or 0x00000000 (false) based on their presence in test_elements.
            Parameters:
                * `elements` (Tensor): Tensor of values to be checked for presence in test_elements. The output will have 0xFFFFFFFF (true) or 0x00000000 (false) at each position depending on whether the corresponding element is present in test_elements (and the invert flag).
                * `test_elements` (Tensor): Tensor containing the values to be checked against. It must have the same dtype as `elements`. If an element from `elements` is present in `test_elements`, the corresponding output value will be 0xFFFFFFFF (unless inverted).
            Keyword Arguments:
                * `invert` (bool): If True, inverts the output so that elements present in test_elements are marked with 0x00000000 and others with 0xFFFFFFFF.
            Notes:
                * `assume_unique` (bool): Currently has no effect, but is reserved for potential future optimizations.
                * Both input tensors must be allocated on a device.
                * Supported dtypes: bfloat16, float32, int32, uint32, uint16 and uint8.
                * The input tensors should be interleaved and in DRAM. Sharded tensors are not supported.
                * If `elements` is empty, nothing runs on the device: the result is an empty uint32 tensor, or `out` if it is given.
        )doc";

    ttnn::bind_function<"isin", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::isin,
        nb::arg("elements").noconvert(),
        nb::arg("test_elements").noconvert(),
        nb::kw_only(),
        nb::arg("assume_unique") = false,
        nb::arg("invert") = false,
        nb::arg("out") = nb::none());
}

}  // namespace ttnn::operations::experimental::isin::detail
