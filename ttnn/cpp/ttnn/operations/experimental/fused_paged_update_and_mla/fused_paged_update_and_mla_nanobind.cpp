// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include "fused_paged_update_and_mla.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::fused_paged_update_and_mla {

void bind_fused_paged_update_and_mla(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::fused_paged_update_and_mla,
        R"doc(
        Fused Paged Update and MLA Kernel.
        Maintains KV update in L1 Ring Buffer for zero-latency access.
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::experimental::fused_paged_update_and_mla)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& kv_cache) -> ttnn::Tensor {
                return self(input_tensor, kv_cache);
            },
            nb::arg("input_tensor"),
            nb::arg("kv_cache")
        }
    );
}

} // namespace ttnn::operations::experimental::fused_paged_update_and_mla
