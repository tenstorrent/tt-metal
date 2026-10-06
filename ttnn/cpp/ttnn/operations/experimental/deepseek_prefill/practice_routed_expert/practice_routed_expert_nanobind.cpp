// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "practice_routed_expert_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "practice_routed_expert.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert::detail {

void bind_practice_routed_expert(nb::module_& mod) {
    // Bound as ttnn.practice_routed_expert, not under ttnn.experimental.deepseek_prefill like its
    // neighbours: that is the name the practice tests call.
    ttnn::bind_function<"practice_routed_expert">(
        mod,
        R"doc(
            One Kimi K3 routed expert as a single fused op, on that expert's tokens only:

                out = situ_glu(x @ w_gate, x @ w_up) @ w_down

            with SiTU-GLU's betas (4 for gate, 25 for up) baked into the kernel. A practice version of
            unified_routed_expert_moe without its dispatch buffer, token counts, region offsets or
            expert-id table.

            Args:
                x (ttnn.Tensor): (T, K) TILE BFLOAT16 or BFLOAT8_B, DRAM interleaved.
                w_gate (ttnn.Tensor): (K, N) TILE, DRAM interleaved, transposed from the HF (out, in) layout.
                w_up (ttnn.Tensor): (K, N), same layout and dtype as w_gate.
                w_down (ttnn.Tensor): (N, K), same layout and dtype as w_gate.

            Keyword Args:
                compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): matmul math fidelity and
                    accumulator. Defaults to HiFi4 with fp32 accumulation.

            Returns:
                ttnn.Tensor: (T, K) TILE in x's dtype, DRAM interleaved.
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::practice_routed_expert::practice_routed_expert,
        nb::arg("x").noconvert(),
        nb::arg("w_gate").noconvert(),
        nb::arg("w_up").noconvert(),
        nb::arg("w_down").noconvert(),
        nb::kw_only(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_practice_routed_expert(::nanobind::module_& mod) {
    practice_routed_expert::detail::bind_practice_routed_expert(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
