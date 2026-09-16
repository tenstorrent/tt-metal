// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mix_streams_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/mix_streams/mix_streams.hpp"

namespace ttnn::operations::experimental::deepseek::mix_streams::detail {

void bind_mix_streams(nb::module_& mod) {
    ttnn::bind_function<"mix_streams", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        Experimental fused hyper-connection stream-mixing ("_mix") step for DeepSeek V4-Flash.

        Replaces the per-token Python sequence in ``DeepSeekV4DecoderLayer._mix``
        (models/experimental/deepseek_v4_flash/tt/decoder_layer.py, lines 97-121) with a
        single fused kernel::

            placement   = post[..,None] * sublayer_out[..,None,:]            [1, T, hc, D]
            mixed       = matmul(comb^T, streams)                            [1, T, hc, D]
            new_streams = (placement + mixed).reshape([B, S, hc, D])

        where ``T == B*S``. Both terms are single-tile matmuls accumulated into the same
        destination register, so the step costs one dispatch instead of four. ``post``,
        ``comb`` and TILE ``streams`` are 32x32; ``sublayer_out`` is ROW_MAJOR (packed as
        1x32 faces). When ``sublayer_out`` or ``streams`` is ROW_MAJOR, dest is untilized
        and the valid hc rows are written as ROW_MAJOR (decode: attention/MoE emit RM
        while the residual is still TILE). TILE-only inputs keep TILE output.
        When D is divisible by 64 with a tile-aligned shard width, the default output
        is WIDTH_SHARDED L1 on 64 cores.
        It runs at HiFi4 with fp32 destination accumulation.
        Shapes the kernel does not cover fall back to the equivalent op sequence.

        Args:
            post (ttnn.Tensor): sublayer-output placement weights, [B, S, hc, 1].
            comb (ttnn.Tensor): doubly-stochastic stream-mixing matrix, [B, S, hc, hc]
                (consumed transposed -- mixed over the FIRST hc axis).
            sublayer_out (ttnn.Tensor): sublayer output for the current token, [B, S, 1, D].
            streams (ttnn.Tensor): residual-stream stack, [B, S, hc, D].

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): output memory config. Defaults to
                WIDTH_SHARDED L1 on 64 cores when D allows; otherwise the ``streams``
                tensor's memory config.
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): matmul compute
                settings. Defaults to HiFi4 / fp32 dest acc / packer-l1-acc (``_HIFI4``).

        Returns:
            ttnn.Tensor: new residual-stream stack, [B, S, hc, D]. ROW_MAJOR when
                ``sublayer_out`` or ``streams`` is ROW_MAJOR, otherwise TILE. WIDTH_SHARDED
                on 64 cores when D allows.
        )doc",
        &ttnn::experimental::deepseek::mix_streams::mix_streams,
        nb::arg("post"),
        nb::arg("comb"),
        nb::arg("sublayer_out"),
        nb::arg("streams"),
        nb::kw_only(),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::mix_streams::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_mix_streams(::nanobind::module_& mod) { mix_streams::detail::bind_mix_streams(mod); }

}  // namespace ttnn::operations::experimental::deepseek::detail
