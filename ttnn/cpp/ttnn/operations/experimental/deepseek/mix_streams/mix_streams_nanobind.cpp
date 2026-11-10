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
        ``comb`` and ``streams`` are TILE 32x32; ``sublayer_out`` is ROW_MAJOR (packed as
        1x32 faces). Output is TILE. It runs at HiFi4 with fp32 destination accumulation.
        Shapes the kernel does not cover fall back to the equivalent op sequence.

        Args:
            post (ttnn.Tensor): sublayer-output placement weights, [B, S, hc, 1].
            comb (ttnn.Tensor): doubly-stochastic stream-mixing matrix, [B, S, hc, hc]
                (consumed transposed -- mixed over the FIRST hc axis).
            sublayer_out (ttnn.Tensor): sublayer output for the current token, [B, S, 1, D].
            streams (ttnn.Tensor): residual-stream stack, [B, S, hc, D].

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): output memory config. Defaults to the
                ``streams`` tensor's memory config.
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): matmul compute
                settings. Defaults to HiFi4 / fp32 dest acc / packer-l1-acc (``_HIFI4``).

        Returns:
            ttnn.Tensor: new residual-stream stack, [B, S, hc, D] (TILE, matching ``streams``).
        )doc",
        &ttnn::experimental::deepseek::mix_streams::mix_streams,
        nb::arg("post"),
        nb::arg("comb"),
        nb::arg("sublayer_out"),
        nb::arg("streams"),
        nb::kw_only(),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt);

    ttnn::bind_function<"stream_mix", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        Hyper-connection stream mixing over a stream-major residual stack (DeepSeek V4.1 prefill).

        ``streams`` is [1, hc, T, D] -- each stream a dense [T, D] slab, so the stack has none of
        the per-token tile padding of the token-major [1, T, hc, D]. With ``comb`` weights it is
        the residual mix; with ``pre`` weights it is the collapse::

            comb [1, T, hc, hc], post [1, T, hc, 1], sublayer_out [1, 1, T, D]:
                out[:, i] = sum_j comb[t, j, i] * streams[:, j] + post[t, i] * sublayer_out   [1, hc, T, D]
            pre [1, T, 1, hc]:
                out[:, 0] = sum_j pre[t, 0, j] * streams[:, j]                               [1, 1, T, D]

        All inputs are BFLOAT16 TILE 32x32 interleaved, hc <= 16 and D a multiple of 32.

        Args:
            streams (ttnn.Tensor): residual-stream stack, [1, hc, T, D].
            weights (ttnn.Tensor): comb [1, T, hc, hc] or pre [1, T, 1, hc].
            post (Optional[ttnn.Tensor]): sublayer-output placement weights, [1, T, hc, 1].
            sublayer_out (Optional[ttnn.Tensor]): sublayer output, [1, 1, T, D]. Given with ``post``.

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): output memory config. Defaults to the
                ``streams`` tensor's memory config.
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): defaults to HiFi4 /
                fp32 dest acc.

        Returns:
            ttnn.Tensor: [1, hc, T, D] for comb weights, [1, 1, T, D] for pre weights (TILE).
        )doc",
        &ttnn::experimental::deepseek::mix_streams::stream_mix,
        nb::arg("streams"),
        nb::arg("weights"),
        nb::arg("post") = std::nullopt,
        nb::arg("sublayer_out") = std::nullopt,
        nb::kw_only(),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::mix_streams::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_mix_streams(::nanobind::module_& mod) { mix_streams::detail::bind_mix_streams(mod); }

}  // namespace ttnn::operations::experimental::deepseek::detail
