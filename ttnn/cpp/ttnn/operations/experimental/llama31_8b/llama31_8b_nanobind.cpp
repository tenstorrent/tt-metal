// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama31_8b_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "argmax_decode/argmax_decode_nanobind.hpp"
#include "concat_heads_decode/concat_heads_decode_nanobind.hpp"
#include "matmul_bh/matmul_bh_nanobind.hpp"
#include "qkv_heads_decode/qkv_heads_decode_nanobind.hpp"
#include "rms_norm_residual/rms_norm_residual_nanobind.hpp"
#include "rms_norm_residual_decode/rms_norm_residual_decode_nanobind.hpp"
#include "rotary_bf8/rotary_bf8_nanobind.hpp"
#include "sdpa_bh/sdpa_bh_nanobind.hpp"

namespace ttnn::experimental::llama31_8b {

void bind_llama31_8b(nb::module_& mod) {
    auto llama31_8b = mod.def_submodule("llama31_8b", "Llama-3.1-8B-specific operations");
    bind_argmax_decode(llama31_8b);
    bind_concat_heads_decode(llama31_8b);
    bind_matmul_bh(llama31_8b);
    bind_qkv_heads_decode(llama31_8b);
    bind_rms_norm_residual(llama31_8b);
    bind_rms_norm_residual_decode(llama31_8b);
    bind_rotary_bf8(llama31_8b);
    bind_sdpa_bh(llama31_8b);
}

}  // namespace ttnn::experimental::llama31_8b
