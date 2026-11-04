// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "router_front_end_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "router_front_end.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

void bind_router_front_end(nb::module_& mod) {
    ttnn::bind_function<"router_front_end", "ttnn.experimental.nomic_embed_text_v2_moe.">(
        mod,
        R"doc(
        An MoE layer's router front end in one pass over its input: each token's probabilities over the
        experts, and the input's rows row-major, the table the routed dispatch gathers from.

        scores = softmax(input @ weight) over the experts, in fp32: the matmul takes K in one block,
        accumulated in the fp32 DST, and the softmax runs ttnn.softmax's numerically stable steps with
        its padding mask for a width off the tile grid, so the scores come out as ttnn.linear and
        ttnn.softmax give them under the same compute config. rows is the input untilized, bit for
        bit. Every core takes whole 32-token blocks; an input block-sharded column-major, as a split
        layer norm leaves it, is read by the cores of the columns holding it. One core reads the weight
        and multicasts it to the others. Nothing is read back to the host, so the call is trace-safe.

        Args:
            input (ttnn.Tensor): (1, 1, tokens, hidden), or (batch, 1, seqlen, hidden) with seqlen a multiple
                of 32 (tokens = batch * seqlen, page for page the same), BFLOAT16, TILE, interleaved or
                sharded (ND included), DRAM or L1: the reader addresses tiles by page id. hidden is whole tiles.
            weight (ttnn.Tensor): (hidden, experts) FLOAT32, TILE, interleaved, leading dimensions 1;
                experts is 1 to 32.

        Keyword Args:
            scores_memory_config (ttnn.MemoryConfig, optional): interleaved. Defaults to DRAM.
            rows_memory_config (ttnn.MemoryConfig, optional): interleaved. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): fp32_dest_acc_en must be
                on. Defaults to HiFi4 with fp32 accumulation, which the softmax needs both of.

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor]: scores (1, 1, tokens, experts) FLOAT32, TILE, the tile's
            padded columns zero; rows (1, 1, tokens, hidden) BFLOAT16, ROW_MAJOR.
        )doc",
        &ttnn::experimental::nomic_embed_text_v2_moe::router_front_end,
        nb::arg("input"),
        nb::arg("weight"),
        nb::kw_only(),
        nb::arg("scores_memory_config") = nb::none(),
        nb::arg("rows_memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
