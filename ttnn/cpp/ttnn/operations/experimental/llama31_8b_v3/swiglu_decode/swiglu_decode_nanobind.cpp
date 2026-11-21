// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_decode_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "swiglu_decode.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::experimental::llama31_8b_v3 {

void bind_swiglu_decode(nb::module_& llama31_8b_v3) {
    const auto* doc = R"doc(
        Decode MLP gate for Llama-3.1-8B on Blackhole P100: ``silu(gate) * up`` written as bfloat8_b in the inputs'
        memory config.

        Output is bit-identical to ``ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat8_b, memory_config=gate.memory_config())``. Call ``swiglu_decode_supported`` first and use
        ``ttnn.mul`` when it returns False.

        Args:
            gate (ttnn.Tensor): [1, 1, 32, 14336] bfloat16, TILE, width sharded in L1 on the first 64 cores (row major).
            up (ttnn.Tensor): same shape, dtype and memory config as ``gate``.

        Returns:
            ttnn.Tensor: [1, 1, 32, 14336] bfloat8_b, TILE, same memory config as ``gate``.
        )doc";

    ttnn::bind_function<"swiglu_decode", "ttnn.experimental.llama31_8b_v3.">(
        llama31_8b_v3, doc, &swiglu_decode, nb::arg("gate"), nb::arg("up"));

    const auto* doc_supported = R"doc(
        True when ``swiglu_decode`` accepts ``gate`` and ``up``: Blackhole 11x10 grid, both [1, 1, 32, 14336] bfloat16
        TILE, width sharded in L1 on the first 64 cores row major with 224-wide shards, same memory config.
        )doc";

    ttnn::bind_function<"swiglu_decode_supported", "ttnn.experimental.llama31_8b_v3.">(
        llama31_8b_v3, doc_supported, &swiglu_decode_supported, nb::arg("gate"), nb::arg("up"));
}

}  // namespace ttnn::experimental::llama31_8b_v3
