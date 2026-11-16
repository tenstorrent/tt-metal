// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama31_8b_v3_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "argmax_decode/argmax_decode_nanobind.hpp"
#include "concat_heads_decode/concat_heads_decode_nanobind.hpp"
#include "kv_rows_update/kv_rows_update_nanobind.hpp"
#include "matmul_bh/matmul_bh_nanobind.hpp"
#include "qkv_heads_decode/qkv_heads_decode_nanobind.hpp"
#include "rms_norm_residual/rms_norm_residual_nanobind.hpp"
#include "rms_norm_residual_decode/rms_norm_residual_decode_nanobind.hpp"
#include "rotary_bf8/rotary_bf8_nanobind.hpp"
#include "sdpa_bh/sdpa_bh_nanobind.hpp"
#include "sdpa_decode_bh/sdpa_decode_bh_nanobind.hpp"
#include "swiglu_decode/swiglu_decode_nanobind.hpp"

namespace ttnn::experimental::llama31_8b_v3 {

void bind_llama31_8b_v3(nb::module_& mod) {
    auto llama31_8b_v3 = mod.def_submodule("llama31_8b_v3", "Llama-3.1-8B-specific operations");
    bind_argmax_decode(llama31_8b_v3);
    bind_concat_heads_decode(llama31_8b_v3);
    bind_kv_rows_update(llama31_8b_v3);
    bind_matmul_bh(llama31_8b_v3);
    bind_qkv_heads_decode(llama31_8b_v3);
    bind_rms_norm_residual(llama31_8b_v3);
    bind_rms_norm_residual_decode(llama31_8b_v3);
    bind_rotary_bf8(llama31_8b_v3);
    bind_sdpa_bh(llama31_8b_v3);
    bind_sdpa_decode_bh(llama31_8b_v3);
    bind_swiglu_decode(llama31_8b_v3);
}

}  // namespace ttnn::experimental::llama31_8b_v3
