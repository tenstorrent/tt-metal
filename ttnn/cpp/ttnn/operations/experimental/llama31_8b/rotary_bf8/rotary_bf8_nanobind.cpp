// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_bf8_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "rotary_bf8.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::experimental::llama31_8b {

void bind_rotary_bf8(nb::module_& llama31_8b) {
    const auto* doc = R"doc(
        Prefill rotary embedding written straight to ``output_dtype``, for Llama-3.1-8B on Blackhole.

        Bit-identical to ``ttnn.typecast(ttnn.experimental.rotary_embedding_llama(input, cos_cache, sin_cache,
        trans_mat, is_decode_mode=False), output_dtype)`` with the same compute kernel config. Shapes outside this
        op's table run exactly that sequence.

        Args:
            input (ttnn.Tensor): [1, n_heads, S, head_dim] bfloat16, TILE, DRAM interleaved.
            cos_cache (ttnn.Tensor): [1, 1, S, head_dim] bfloat16, TILE, DRAM interleaved.
            sin_cache (ttnn.Tensor): same as cos_cache.
            trans_mat (ttnn.Tensor): [1, 1, 32, 32] bfloat16, TILE, DRAM interleaved.

        Keyword Args:
            output_dtype (ttnn.DataType): defaults to bfloat8_b.
            memory_config (ttnn.MemoryConfig, optional): defaults to the input's.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): defaults to rotary_embedding_llama's.

        Returns:
            ttnn.Tensor: the rotated input in output_dtype.
        )doc";

    ttnn::bind_function<"rotary_bf8", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        doc,
        &rotary_bf8,
        nb::arg("input"),
        nb::arg("cos_cache"),
        nb::arg("sin_cache"),
        nb::arg("trans_mat"),
        nb::kw_only(),
        nb::arg("output_dtype") = DataType::BFLOAT8_B,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::experimental::llama31_8b
