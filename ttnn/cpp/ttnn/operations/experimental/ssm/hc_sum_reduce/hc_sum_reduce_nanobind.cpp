// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hc_sum_reduce_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "hc_sum_reduce.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::ssm::detail {

namespace nb = nanobind;

void bind_hc_sum_reduce(nb::module_& mod) {
    const auto* const doc = R"doc(
        Performs a custom reduction along dim 3 which is used in the SSM block of the Mamba architecture. Performs the following PyTorch equivalent (where latent_size = 32):
            x = torch.sum(x.reshape(1, 1, shape[2], shape[3] // latent_size, latent_size), dim=-1).reshape(1, 1, shape[2], shape[3] // latent_size)
    )doc";

    ttnn::bind_function<"hc_sum_reduce", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::hc_sum_reduce,
        nb::arg("input"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("math_fidelity") = nb::none());
}

}  // namespace ttnn::operations::experimental::ssm::detail
