// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_heads_decode_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "concat_heads_decode.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::experimental::llama31_8b {

void bind_concat_heads_decode(nb::module_& llama31_8b) {
    const auto* doc = R"doc(
        Concatenates the decode attention heads, for Llama-3.1-8B decode on Blackhole.

        Reads the SDPA decode output from DRAM and returns the same tensor as
        ``ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(input, <height-sharded SDPA output>), num_heads)``:
        [1, 1, 32, num_heads * head_dim], L1 width-sharded, one head per core. Row 0 (the batch-1 user) is bit-identical.

        Args:
            input (ttnn.Tensor): [1, 1, 32, 128] bfloat16, TILE, DRAM interleaved.
            num_heads (int): 32.

        Returns:
            ttnn.Tensor: the concatenated heads.
        )doc";

    ttnn::bind_function<"concat_heads_decode", "ttnn.experimental.llama31_8b.">(
        llama31_8b, doc, &concat_heads_decode, nb::arg("input"), nb::arg("num_heads"));
}

}  // namespace ttnn::experimental::llama31_8b
