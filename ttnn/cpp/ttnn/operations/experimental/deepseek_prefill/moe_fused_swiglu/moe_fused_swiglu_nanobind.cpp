// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_fused_swiglu_nanobind.hpp"

#include <nanobind/stl/optional.h>

#include "moe_fused_swiglu.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::detail {

void bind_moe_fused_swiglu(nb::module_& mod) {
    ttnn::bind_function<"moe_fused_swiglu", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Fused routed-expert SwiGLU using a configurable rectangular worker grid.

        Computes a gated activation over ``activations @ w_gate`` and
        ``activations @ w_up``, followed by ``@ w_down``, in one device program.
        ``activation`` selects plain SiLU SwiGLU (default) or Kimi K3 SiTU-GLU.
        The number of valid token rows is read on device from
        ``counts[global_expert_idx_table[local_expert_id]]``; no host readback
        or host branch depends on the count.

        The input activations must be BFLOAT16 ROW_MAJOR or BFLOAT8_B TILE. All weights
        must be TILE and share BFLOAT4_B, BFLOAT8_B, or BFLOAT16 dtype. The
        default output is BFLOAT8_B TILE in DRAM. By default all cores in the
        device's compute-with-storage grid are used. ``core_grid=(x, y)`` is an
        explicit rectangular-prefix override.
        )doc",
        &moe_fused_swiglu,
        nb::arg("input").noconvert(),
        nb::arg("w_gate").noconvert(),
        nb::arg("w_up").noconvert(),
        nb::arg("w_down").noconvert(),
        nb::arg("counts").noconvert(),
        nb::arg("global_expert_idx_table").noconvert(),
        nb::arg("local_expert_id"),
        nb::kw_only(),
        nb::arg("input_m_tiles") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("core_grid") = nb::none(),
        nb::arg("output") = nb::none(),
        nb::arg("expert_region_offsets") = nb::none(),
        nb::arg("read_x_at_offset") = false,
        nb::arg("activation") = RoutedExpertActivation::Silu);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_moe_fused_swiglu(nb::module_& mod) { moe_fused_swiglu::detail::bind_moe_fused_swiglu(mod); }
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
