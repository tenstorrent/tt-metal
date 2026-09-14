// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "gdn_spec_tloop_proto_nanobind.hpp"

#include "gdn_spec_tloop_proto.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::gdn_spec_tloop_proto::detail {

void bind_gdn_spec_tloop_proto(nb::module_& mod) {
    ttnn::bind_function<"gdn_spec_tloop_proto", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        M0a SCRATCH prototype: row-batched T-loop gated-delta-rule recurrence, one core per (user, value head).
        qkv [1, R <= 32, W] BFLOAT16 TILE holds rows u*T + t of the (post-conv) [q | k | v | z | a | b] projection;
        ring [T*B*Nv, Dk, Dv] FLOAT32 is read at block (s0_slot*B*Nv + u*Nv + h) and written in place at blocks
        (t*B*Nv + u*Nv + h) for t < T. Returns [1, R, Nv*Dv] (rows u*T + t valid).
        row_batched: l2norms / gates / kt / rmsnorm / output gate once per core for the T rows (else per token).
        write_ring: False skips the per-token ring writes (ablation).
        opt_flags: bit0 two-phase reader, bit1 dual-pack hnew, bit2 writer-only mode, bit3 single-row output writes.
        )doc",
        &ttnn::experimental::kda::gdn_spec_tloop_proto,
        nb::arg("qkv").noconvert(),
        nb::arg("dt_bias").noconvert(),
        nb::arg("neg_exp_A").noconvert(),
        nb::arg("ring").noconvert(),
        nb::arg("weight").noconvert(),
        nb::arg("num_value_heads"),
        nb::arg("num_key_heads"),
        nb::arg("key_dim"),
        nb::arg("value_dim"),
        nb::arg("T"),
        nb::arg("B"),
        nb::arg("qkvz_dim"),
        nb::kw_only(),
        nb::arg("s0_slot") = 0,
        nb::arg("scale") = nb::none(),
        nb::arg("l2_epsilon") = 1e-6f,
        nb::arg("norm_epsilon") = 1e-6f,
        nb::arg("row_batched") = true,
        nb::arg("write_ring") = true,
        nb::arg("opt_flags") = 0,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_dtype") = ttnn::DataType::BFLOAT16);
}

}  // namespace ttnn::operations::experimental::kda::gdn_spec_tloop_proto::detail
