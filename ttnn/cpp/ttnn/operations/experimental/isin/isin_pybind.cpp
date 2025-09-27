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
            This operator returns a uint32 tensor of the same specification as
            the elements tensor (dtype, shape, layout, memory config) with those of the elements filled out with 0xFFFFFFFF (fully filled sized bitmask)
            value (this rule can be inverted with the invert flag) that correspond
            to those of the values from the elements tensor that are contained in the
            test_elements tensor - in other words, this operator returns a tensor like input tensor containing answers to "does this element belong to anywhere inside test_elements tensor?".

            Parameters:
                * `elements` (Tensor): integers to be masked with 0s or 1s depending no their existence in test_elements and the invert flag
                * `test_elements` (Tensor): all integers for which the output tensor should have 0xFFFFFFFF as the corresponding output value

            Keyword Arguments:
                * `invert` (bool): invert nonzero output with zeroes and vice versa

            Notes:
                * `assume_unique` (bool): does nothing for the time being, but is reserved for potential future optimizations
                * The input tensors should be interleaved and in DRAM
                * Both input tensors can be of any specification

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
