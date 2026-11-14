// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_norm_residual_decode_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "rms_norm_residual_decode.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::experimental::llama31_8b {

void bind_rms_norm_residual_decode(nb::module_& llama31_8b) {
    const auto* doc = R"doc(
        Residual add and sharded RMSNorm in one program, for Llama-3.1-8B decode on a Blackhole P100.

        Computes ``h = input + residual`` and returns ``ttnn.rms_norm(h, epsilon=epsilon, weight=weight,
        program_config=program_config, memory_config=memory_config)``. ``h`` is written back into
        ``residual`` in place, in its own sharded layout. Both results are bit-identical to resharding
        ``input`` and ``residual`` to ``memory_config``, ``ttnn.add``, and ``ttnn.rms_norm`` with the same
        program and compute kernel configs. Only the layouts of its table are accepted: W = 4096, residual
        width-sharded [32, 128] on 8x4 cores, input on that layout or [32, 64] on 8x8 cores, and output with
        program_config either 8x4 / [32, 128] or 8x8 / [32, 64]. ``rms_norm_residual_decode_supported``
        tells whether a call is in the table.

        Args:
            input (ttnn.Tensor): [1, 1, 32, 4096] bfloat16, TILE, L1 width sharded.
            residual (ttnn.Tensor): same shape as ``input``, bfloat16, TILE, L1 width sharded.
            weight (ttnn.Tensor): RMSNorm weight, bfloat16 ROW_MAJOR [1, 1, 128, 32] in DRAM.
            epsilon (float): RMSNorm epsilon.

        Keyword Args:
            program_config (ttnn.LayerNormShardedMultiCoreProgramConfig): one of the two table configs.
            memory_config (ttnn.MemoryConfig): the L1 width-sharded output layout of that config.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig): compute kernel config.

        Returns:
            ttnn.Tensor: the normalized output.
        )doc";

    ttnn::bind_function<"rms_norm_residual_decode", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        doc,
        &rms_norm_residual_decode,
        nb::arg("input"),
        nb::arg("residual"),
        nb::arg("weight"),
        nb::arg("epsilon"),
        nb::kw_only(),
        nb::arg("program_config"),
        nb::arg("memory_config"),
        nb::arg("compute_kernel_config"));

    ttnn::bind_function<"rms_norm_residual_decode_supported", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        R"doc(
        True when ``ttnn.experimental.llama31_8b.rms_norm_residual_decode`` accepts this call (its table).
        When False, use ``ttnn.add`` followed by ``ttnn.rms_norm``.
        )doc",
        &rms_norm_residual_decode_supported,
        nb::arg("input"),
        nb::arg("residual"),
        nb::arg("weight"),
        nb::kw_only(),
        nb::arg("program_config"),
        nb::arg("memory_config"));
}

}  // namespace ttnn::experimental::llama31_8b
