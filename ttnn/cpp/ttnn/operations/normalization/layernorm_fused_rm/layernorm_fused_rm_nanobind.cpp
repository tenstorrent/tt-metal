// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_fused_rm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "layernorm_fused_rm.hpp"

namespace ttnn::operations::layernorm_fused_rm {

void bind_layernorm_fused_rm_operation(nb::module_& mod) {
    const auto doc =
        R"doc(LayerNorm operation that accepts row-major input and produces row-major output. Normalizes each row by computing mean and variance across the last dimension, then applies learnable affine transformation (gamma and beta).)doc";

    bind_registered_operation(
        mod,
        ttnn::layernorm_fused_rm,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("gamma"),
            nb::arg("beta"),
            nb::arg("epsilon") = 1e-5f,
            nb::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::layernorm_fused_rm
