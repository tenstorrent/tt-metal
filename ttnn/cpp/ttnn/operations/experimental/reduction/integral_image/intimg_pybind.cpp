// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "intimg_pybind.hpp"
#include "intimg.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::reduction::detail {
void bind_reduction_intimg_operation(py::module& module) {
    auto docstring =
        R"doc(
        )doc";

    using OperationType = decltype(ttnn::experimental::intimg);
    bind_registered_operation(
        module,
        ttnn::experimental::intimg,
        docstring,
        ttnn::pybind_overload_t{
            [](const OperationType& self, const ttnn::Tensor& input_tensor) -> Tensor { return self(input_tensor); },
            py::arg("input_tensor").noconvert(),
            py::kw_only()});
}

}  // namespace ttnn::operations::experimental::reduction::detail
