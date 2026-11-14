// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_decode_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "argmax_decode.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::experimental::llama31_8b {

void bind_argmax_decode(nb::module_& llama31_8b) {
    const auto* doc = R"doc(
        Greedy decode argmax over the last dim of width-sharded TILE logits.

        Output is bit-identical to ``ttnn.argmax(ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=False)``:
        NaN is the maximum, -0 equals +0, ties take the first index. Call ``argmax_decode_supported`` first and use
        ``ttnn.argmax`` when it returns False.

        Args:
            input (ttnn.Tensor): [1, 1, 32, V] bfloat16, TILE, width sharded in L1 as listed in the op's table.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): interleaved output memory config, defaults to DRAM.
            output_tensor (ttnn.Tensor, optional): preallocated [1, 1, 32] uint32 ROW_MAJOR output.

        Returns:
            ttnn.Tensor: [1, 1, 32] uint32, ROW_MAJOR.
        )doc";

    ttnn::bind_function<"argmax_decode", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        doc,
        &argmax_decode,
        nb::arg("input"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());

    const auto* doc_supported = R"doc(
        True when ``argmax_decode`` accepts these arguments: Blackhole, [1, 1, 32, V] bfloat16 TILE, width sharded in L1
        with a shape and core grid from the op's table; ``memory_config`` interleaved; ``output_tensor``, if given,
        an allocated [1, 1, 32] uint32 ROW_MAJOR interleaved tensor on the input's device.
        )doc";

    ttnn::bind_function<"argmax_decode_supported", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        doc_supported,
        &argmax_decode_supported,
        nb::arg("input"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::experimental::llama31_8b
