// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_rotate_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn/operations/experimental/fused_rotate/fused_rotate.hpp"
#include "ttnn/operations/experimental/fused_rotate/fused_rotate_gc.hpp"
#include "ttnn/operations/experimental/fused_rotate/fused_ln_bw.hpp"
#include "ttnn/operations/experimental/fused_rotate/fused_gate.hpp"

namespace ttnn::operations::experimental::fr_detail {

void bind_fused_rotate(nb::module_& mod) {
    mod.def(
        "fused_ln_bw",
        [](const ttnn::Tensor& gy,
           const ttnn::Tensor& x,
           const ttnn::Tensor& red,
           const ttnn::Tensor& n,
           const ttnn::Tensor& gamma,
           uint32_t W,
           uint32_t eps_bits) {
            return ttnn::operations::experimental::fused_ln_bw(gy, x, red, n, gamma, W, eps_bits);
        },
        nb::arg("gy").noconvert(),
        nb::arg("x").noconvert(),
        nb::arg("red").noconvert(),
        nb::arg("n").noconvert(),
        nb::arg("gamma").noconvert(),
        nb::arg("W"),
        nb::arg("eps_bits"),
        "Fused SiLU-bw + LayerNorm-bw: gy=g_out*silu'(n)*gamma in-kernel, then dx.");
    mod.def(
        "fused_gate",
        [](const ttnn::Tensor& a,
           const ttnn::Tensor& gate,
           const ttnn::Tensor& b,
           uint32_t Wt,
           uint32_t Gt,
           uint32_t Ht,
           uint32_t mode) { return ttnn::operations::experimental::fused_gate(a, gate, b, Wt, Gt, Ht, mode); },
        nb::arg("a").noconvert(),
        nb::arg("gate").noconvert(),
        nb::arg("b").noconvert(),
        nb::arg("Wt"),
        nb::arg("Gt"),
        nb::arg("Ht"),
        nb::arg("mode"),
        "Fused SO(3) gate: out=[silu(a[:H]) | a[H:]*gate] (mode0 fwd) or [a[:H]*silu'(b) | a[H:]*gate] (mode1 bw).");
    mod.def(
        "fused_rotate_gc",
        [](const ttnn::Tensor& gout,
           const ttnn::Tensor& xin,
           const ttnn::Tensor& sel,
           uint32_t n_out,
           uint32_t n_in,
           uint32_t W,
           const std::vector<uint32_t>& is_,
           const std::vector<uint32_t>& js) {
            return ttnn::operations::experimental::fused_rotate_gc(gout, xin, sel, n_out, n_in, W, is_, js);
        },
        nb::arg("gout").noconvert(),
        nb::arg("xin").noconvert(),
        nb::arg("sel").noconvert(),
        nb::arg("n_out"),
        nb::arg("n_in"),
        nb::arg("W"),
        nb::arg("is_"),
        nb::arg("js"),
        "Fused per-edge coefficient adjoint (rotate_bw dE/dcoef) mul+reduce+place in one kernel.");
    mod.def(
        "fused_rotate",
        [](const ttnn::Tensor& x_flat,
           const ttnn::Tensor& coef_exp,
           uint32_t n_in,
           uint32_t n_out,
           uint32_t W,
           const std::vector<uint32_t>& deg,
           const std::vector<uint32_t>& ks,
           const std::vector<uint32_t>& js) {
            return ttnn::operations::experimental::fused_rotate(x_flat, coef_exp, n_in, n_out, W, deg, ks, js);
        },
        nb::arg("x_flat").noconvert(),
        nb::arg("coef_exp").noconvert(),
        nb::arg("n_in"),
        nb::arg("n_out"),
        nb::arg("W"),
        nb::arg("deg"),
        nb::arg("ks"),
        nb::arg("js"),
        "Fused per-edge sparse Wigner rotation (all nnz MACs in one kernel launch).");
}

}  // namespace ttnn::operations::experimental::fr_detail
