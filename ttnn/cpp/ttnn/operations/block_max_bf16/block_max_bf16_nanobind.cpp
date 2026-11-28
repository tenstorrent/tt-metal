// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "block_max_bf16_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "block_max_bf16.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::block_max_bf16 {

// The op refuses only through TT_FATAL (raised as RuntimeError), so there is no support-refusal translator.
void bind_block_max_bf16_operation(nb::module_& mod) {
    ttnn::bind_function<"block_max_bf16">(
        mod,
        R"doc(
        Block maxima to bf16, for the exact block pre-selected top-k (deepseek_v41_flash_d_p PB.BM).

        Reduce (``strip_major=False``): block_max [.., R, NI] float32 ROW_MAJOR (bf16 values widened) ->
        [.., R, NI / factor] bfloat16 ROW_MAJOR, column b = the max of input columns [b factor, (b + 1) factor)
        (exact), for the first ``n`` output columns (rounded up to 8; the rest not written).

        Strip-major (``strip_major=True``, PB.FBM): block_max [.., S NB / 8, 256] bfloat16 ROW_MAJOR, strip-major in
        512-byte pages of 8 blocks -> ``output_tensor`` [.., 32 S, NB] bfloat16 ROW_MAJOR (required: it gives S and
        NB; a transpose of each strip, exact), the first ``n`` blocks (rounded up to 8).

        Args:
            block_max (ttnn.Tensor): the input, DRAM interleaved ROW_MAJOR.

        Keyword Args:
            factor (int): a power of two <= 16 (1 with strip_major). Defaults to 1.
            n (int, optional): output columns / blocks, a runtime value (one program for every n).
            n_tensor (ttnn.Tensor, optional): [1] uint32 ROW_MAJOR DRAM tensor holding n, read on the device.
                Exactly one of n / n_tensor.
            output_tensor (ttnn.Tensor, optional): preallocated output (required with strip_major).
            memory_config (ttnn.MemoryConfig, optional): output placement; defaults to DRAM interleaved.
            max_grid (ttnn.CoreCoord or tuple[int, int], optional): cap on the worker grid (x, y); defaults to the
                whole grid.
            strip_major (bool): the strip-major transpose instead of the reduce. Defaults to False.

        Returns:
            ttnn.Tensor: the output (``output_tensor`` when given).

        Raises:
            RuntimeError: an input outside the operation's contract.
        )doc",
        &ttnn::block_max_bf16,
        nb::arg("block_max"),
        nb::kw_only(),
        nb::arg("factor") = 1,
        nb::arg("n") = nb::none(),
        nb::arg("n_tensor") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("max_grid") = nb::none(),
        nb::arg("strip_major") = false);
}

}  // namespace ttnn::operations::block_max_bf16
