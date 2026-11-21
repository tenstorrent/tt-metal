// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_norm_residual_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "rms_norm_residual.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::experimental::llama31_8b {

void bind_rms_norm_residual(nb::module_& llama31_8b) {
    const auto* doc = R"doc(
        Residual add and RMSNorm in one program, for Llama-3.1-8B prefill on Blackhole.

        Returns ``(ttnn.rms_norm(input + residual, epsilon=epsilon, weight=weight), input + residual)``, both
        bit-identical to ``ttnn.add`` followed by ``ttnn.rms_norm`` with the same compute kernel config.

        Args:
            input (ttnn.Tensor): [1, 1, S, W] bfloat16, TILE, DRAM interleaved.
            residual (ttnn.Tensor): same shape, bfloat16 or bfloat8_b, TILE, DRAM interleaved.
            weight (ttnn.Tensor): bfloat16 ROW_MAJOR [1, 1, W / 32, 32].
            epsilon (float): RMSNorm epsilon.

        Keyword Args:
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): must have fp32_dest_acc_en.
                Defaults to HiFi2, fp32 dest accumulation, packer L1 accumulation.

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor]: the normalized output and input + residual.
        )doc";

    ttnn::bind_function<"rms_norm_residual", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        doc,
        &rms_norm_residual,
        nb::arg("input"),
        nb::arg("residual"),
        nb::arg("weight"),
        nb::arg("epsilon"),
        nb::kw_only(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"rms_norm_residual_supported", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        R"doc(
        True when ``ttnn.experimental.llama31_8b.rms_norm_residual`` runs for these tensors: the shape is in its
        table and the free L1 on the device holds its buffers. When False, use ``ttnn.add`` followed by
        ``ttnn.rms_norm``.
        )doc",
        &rms_norm_residual_supported,
        nb::arg("input"),
        nb::arg("residual"));
}

}  // namespace ttnn::experimental::llama31_8b
