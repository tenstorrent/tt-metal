// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_bh_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "matmul_bh.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b {

void bind_matmul_bh(nb::module_& llama31_8b) {
    ttnn::bind_function<"matmul_bh", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        R"doc(
        Llama-3.1-8B's matmuls on a Blackhole p100a, with programs built for that card. Takes the arguments the
        model passes to ttnn.linear and returns the same result; calls it does not cover raise, so check
        matmul_bh_covers first and use ttnn.linear otherwise.
        )doc",
        &matmul_bh,
        nb::arg("input_tensor_a").noconvert(),
        nb::arg("input_tensor_b").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"matmul_bh_covers", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        R"doc(
        True when ttnn.experimental.llama31_8b.matmul_bh covers this call.
        )doc",
        &matmul_bh_covers,
        nb::arg("input_tensor_a").noconvert(),
        nb::arg("input_tensor_b").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"minimal_matmul_bh", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        R"doc(
        Llama-3.1-8B's prefill matmuls that the model sends to ttnn.experimental.minimal_matmul, on a Blackhole
        p100a. Calls it does not cover raise, so check minimal_matmul_bh_covers first.
        )doc",
        &minimal_matmul_bh,
        nb::arg("input_tensor").noconvert(),
        nb::arg("weight_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"minimal_matmul_bh_covers", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        R"doc(
        True when ttnn.experimental.llama31_8b.minimal_matmul_bh covers this call.
        )doc",
        &minimal_matmul_bh_covers,
        nb::arg("input_tensor").noconvert(),
        nb::arg("weight_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"qkv_heads_bh", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        R"doc(
        Llama-3.1-8B's prefill QKV projection on a Blackhole p100a, returning (Q, K, V) in the layout
        ttnn.experimental.nlp_create_qkv_heads gives for the fused projection output. Calls it does not cover raise,
        so check qkv_heads_bh_covers first.
        )doc",
        &qkv_heads_bh,
        nb::arg("input_tensor").noconvert(),
        nb::arg("weight_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("num_heads"),
        nb::arg("num_kv_heads"));

    ttnn::bind_function<"qkv_heads_bh_covers", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        R"doc(
        True when ttnn.experimental.llama31_8b.qkv_heads_bh covers this call.
        )doc",
        &qkv_heads_bh_covers,
        nb::arg("input_tensor").noconvert(),
        nb::arg("weight_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("num_heads"),
        nb::arg("num_kv_heads"));
}

}  // namespace ttnn::experimental::llama31_8b
