// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isin_pybind.hpp"

#include "isin.hpp"

#include <optional>

namespace ttnn::operations::experimental::isin::detail {

using namespace ttnn;

void bind_isin_operation(py::module& module) {
    auto doc =
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

    using OperationType = decltype(ttnn::experimental::isin);
    bind_registered_operation(
        module,
        ttnn::experimental::isin,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& elements,
               const Tensor& test_elements,
               bool assume_unique,
               bool invert,
               const std::optional<Tensor>& optional_out) -> Tensor {
                return self(elements, test_elements, assume_unique, invert, optional_out);
            },
            py::arg("elements").noconvert(),
            py::arg("test_elements").noconvert(),
            py::kw_only(),
            py::arg("assume_unique") = false,
            py::arg("invert") = false,
            py::arg("out") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::isin::detail
