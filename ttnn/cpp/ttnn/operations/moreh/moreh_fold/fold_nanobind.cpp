// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/moreh/moreh_fold/fold.hpp"

namespace ttnn::operations::moreh::moreh_fold {
void bind_moreh_fold_operation(nb::module_& mod) {
    ttnn::bind_function<"moreh_fold">(
        mod,
        "Moreh Fold Operation",
        ttnn::overload_t(
            &ttnn::moreh_fold,
            nb::arg("input"),
            nb::arg("output") = nb::none(),
            nb::arg("output_size"),
            nb::arg("kernel_size"),
            nb::arg("dilation") = std::vector<uint32_t>{1, 1},
            nb::arg("padding") = std::vector<uint32_t>{0, 0},
            nb::arg("stride") = std::vector<uint32_t>{1, 1},
            nb::arg("memory_config") = nb::none()));
}
}  // namespace ttnn::operations::moreh::moreh_fold
