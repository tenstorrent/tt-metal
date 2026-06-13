// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "nemotron3_mamba2_decode_owned_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/nemotron3_mamba2_decode_owned/nemotron3_mamba2_decode_owned.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::nemotron3_mamba2_decode_owned::detail {

void bind_nemotron3_mamba2_decode_owned(nb::module_& mod) {
    ttnn::bind_function<"nemotron3_mamba2_decode_owned", "ttnn.experimental.">(
        mod,
        R"doc(
            Owned Mamba2 SSD decode-step kernel for Nemotron-3 Nano.

            Implements one per-token decode step of the Mamba2 selective
            state-space recurrence. Mutates `ssm_state` in place and returns
            (ssm_state, y). `z` is passed in but not consumed inside the
            kernel — the downstream MambaRMSNormGated(y, z) applies it.

            Shapes (Nemotron-3 Nano):
                x, z         : [B, num_heads=64, head_dim=64]   bf16
                dt           : [B, num_heads=64]                bf16
                dt_bias      : [num_heads=64]                   bf16  (weight)
                A_log        : [num_heads=64]                   bf16  (weight)
                D            : [num_heads=64]                   bf16  (weight)
                B_in, C_in   : [B, n_groups=8, ssm_state=128]   bf16
                ssm_state    : [B, num_heads=64, head_dim=64, ssm_state=128] fp32

            Use `debug_mode` to validate incrementally:
                0 = production (full SSD math)
                1 = fill_one smoke (output all 1.0)
                2 = decay * state only
                3 = state correct (decay*state + dt_eff*B*x outer)
                4 = + D*x skip
                5 = + C·state reduce (production equivalent)

            See research/mm7_g1_mamba2_kernel_design.md for the full design.
        )doc",
        &ttnn::experimental::nemotron3_mamba2_decode_owned,
        nb::arg("x").noconvert(),
        nb::arg("z").noconvert(),
        nb::arg("dt").noconvert(),
        nb::arg("dt_bias").noconvert(),
        nb::arg("A_log").noconvert(),
        nb::arg("D").noconvert(),
        nb::arg("B_in").noconvert(),
        nb::arg("C_in").noconvert(),
        nb::arg("ssm_state").noconvert(),
        nb::kw_only(),
        nb::arg("debug_fill").noconvert() = false,
        nb::arg("debug_mode").noconvert() = 0,
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("preallocated_y") = nb::none());
}

}  // namespace ttnn::operations::experimental::nemotron3_mamba2_decode_owned::detail
