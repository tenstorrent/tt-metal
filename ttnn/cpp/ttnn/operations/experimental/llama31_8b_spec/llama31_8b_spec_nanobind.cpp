// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama31_8b_spec_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "argmax_decode/argmax_decode_nanobind.hpp"
#include "concat_heads_decode/concat_heads_decode_nanobind.hpp"
#include "matmul_bh/matmul_bh_nanobind.hpp"
#include "qkv_heads_decode/qkv_heads_decode_nanobind.hpp"
#include "rms_norm_residual/rms_norm_residual_nanobind.hpp"
#include "rms_norm_residual_decode/rms_norm_residual_decode_nanobind.hpp"
#include "rotary_bf8/rotary_bf8_nanobind.hpp"
#include "sdpa_bh/sdpa_bh_nanobind.hpp"
#include "swiglu_decode/swiglu_decode_nanobind.hpp"

namespace ttnn::experimental::llama31_8b_spec {

void bind_llama31_8b_spec(nb::module_& mod) {
    auto llama31_8b_spec = mod.def_submodule("llama31_8b_spec", "Llama-3.1-8B-specific operations");
    bind_argmax_decode(llama31_8b_spec);
    bind_concat_heads_decode(llama31_8b_spec);
    bind_matmul_bh(llama31_8b_spec);
    bind_qkv_heads_decode(llama31_8b_spec);
    bind_rms_norm_residual(llama31_8b_spec);
    bind_rms_norm_residual_decode(llama31_8b_spec);
    bind_rotary_bf8(llama31_8b_spec);
    bind_sdpa_bh(llama31_8b_spec);
    bind_swiglu_decode(llama31_8b_spec);
}

}  // namespace ttnn::experimental::llama31_8b_spec
