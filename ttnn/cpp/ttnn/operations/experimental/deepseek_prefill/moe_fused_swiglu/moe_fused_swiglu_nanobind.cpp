// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_fused_swiglu_nanobind.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "moe_fused_swiglu.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::detail {

void bind_moe_fused_swiglu(nb::module_& mod) {
    ttnn::bind_function<"moe_fused_swiglu", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Fused routed-expert SwiGLU over ALL local experts using a configurable
        rectangular worker grid.

        Takes the dispatched buffer plus one weight triple per local expert and
        runs every expert in ONE device program: the reader, compute and writer
        kernels loop the experts, computing a gated activation over
        ``activations @ w_gate`` and ``activations @ w_up``, followed by
        ``@ w_down``. ``activation`` selects plain SiLU SwiGLU (default) or
        Kimi K3 SiTU-GLU. Each expert's valid token-row count is read on device
        from ``counts[global_expert_idx_table[e]]``; no host readback or host
        branch depends on the count, and no per-expert dispatch happens.

        An expert whose count is zero is skipped uniformly across the grid, so a
        masked ``counts`` vector routes a subset of the experts to this op and
        leaves the rest to another.

        The input activations must be BFLOAT16 ROW_MAJOR or BFLOAT8_B TILE. All weights
        must be TILE and share BFLOAT4_B, BFLOAT8_B, or BFLOAT16 dtype, and every
        expert's weights must share the layout of expert 0 -- one accessor layout
        descriptor per role serves the whole loop. The default output is
        BFLOAT8_B TILE in DRAM. By default all cores in the device's
        compute-with-storage grid are used. ``core_grid=(x, y)`` is an explicit
        rectangular-prefix override.
        )doc",
        &moe_fused_swiglu,
        nb::arg("input").noconvert(),
        nb::arg("w_gates").noconvert(),
        nb::arg("w_ups").noconvert(),
        nb::arg("w_downs").noconvert(),
        nb::arg("counts").noconvert(),
        nb::arg("global_expert_idx_table").noconvert(),
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
