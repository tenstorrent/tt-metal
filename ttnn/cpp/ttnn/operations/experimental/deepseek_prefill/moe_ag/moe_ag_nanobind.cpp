// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ag_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "moe_ag.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_ag::detail {

namespace nb = nanobind;

void bind_moe_ag(nb::module_& mod) {
    ttnn::bind_function<"moe_ag_route_plan", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        All-gather MoE route plan (one program on an 8 x 8 core rectangle). From the dispatch group's gathered top-k
        expert ids and this chip's local-slot map, builds the flat expert space on device.

        Args:
            topk_indices (ttnn.Tensor): gathered top-k [.., T, K] uint16 ROW_MAJOR DRAM (K <= 32).
            local_slot_map (ttnn.Tensor): per device [.., 1, NG] uint32 ROW_MAJOR: global expert id -> local slot
                (< experts_per_chip) or 0xFFFFFFFF.
            experts_per_chip (int): local experts (<= 64).
            num_rows (int): rows of the flat expert space, a multiple of 32 and at least the worst case
                roundup(P, 32) + 32 (min(P, experts_per_chip) - 1) with P = T min(K, experts_per_chip) (no pair is dropped).

        Keyword Args:
            outputs (List[ttnn.Tensor], optional): preallocated (counts, regions, token_index, y_slot).

        Returns:
            [counts [1, NG], regions [1, NG], token_index [1, num_rows], y_slot [1, T K]] uint32 ROW_MAJOR DRAM:
            tokens per global expert, each local expert's first flat row (32-row aligned, local order), flat row ->
            gathered token (region tile tails zero), per (token, k) the flat row of its expert output or 0xFFFFFFFF.
        )doc",
        &moe_ag_route_plan,
        nb::arg("topk_indices").noconvert(),
        nb::arg("local_slot_map").noconvert(),
        nb::arg("experts_per_chip"),
        nb::arg("num_rows"),
        nb::kw_only(),
        nb::arg("outputs") = nb::none());

    ttnn::bind_function<"moe_ag_local_reduce", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        All-gather MoE local reduce: partial[g] = sum over this chip's local experts of w[g, k] * y[y_slot[g, k]]
        (fp32 DEST) for the column's T tokens, over the whole worker grid.

        Args:
            y (ttnn.Tensor): the experts' outputs at their flat rows [.., rows, H] bf16 ROW_MAJOR DRAM (H % 32 == 0).
            y_slot (ttnn.Tensor): [1, T K] uint32 (moe_ag_route_plan).
            weights (ttnn.Tensor): the gathered top-k weights [.., T, K] bf16 ROW_MAJOR.
            chip_info (ttnn.Tensor): per device [.., 1, 16] uint32: word 0 the mesh row, 1 the other row's block start
                (two rows: (1 - row) S), 2 this row's (row S).
            chunk_size_per_chip (int): S (T is a multiple of S).

        Keyword Args:
            phase (int): 0 one pass over the T tokens; 1 / 2 (two mesh rows, fused send-back): the other row's tokens
                into ``other`` [S, H] / this row's tokens plus the peer's gathered phase-1 partial into ``own``.
            split (bool): phase 0, two mesh rows: outputs own [S, H] (this row's tokens) and other [S, H].
            tiled (bool): phase 0: the [T, H] partials as bf16 tiles (the > 2-row reduce-scatter input).
            peer (ttnn.Tensor, optional): phase 2: the gathered phase-1 partials [.., 2 S, H] bf16 ROW_MAJOR.
            outputs (List[ttnn.Tensor], optional): preallocated outputs.

        Returns:
            List[ttnn.Tensor]: [own] ([own, other] when split; [other] for phase 1), bf16 [1, 1, rows, H].
        )doc",
        &moe_ag_local_reduce,
        nb::arg("y").noconvert(),
        nb::arg("y_slot").noconvert(),
        nb::arg("weights").noconvert(),
        nb::arg("chip_info").noconvert(),
        nb::arg("chunk_size_per_chip"),
        nb::kw_only(),
        nb::arg("phase") = 0,
        nb::arg("split") = false,
        nb::arg("tiled") = false,
        nb::arg("peer") = nb::none(),
        nb::arg("outputs") = nb::none());

    ttnn::bind_function<"moe_ag_sum_rows_tiled", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        out[r] = sum_{i < num_blocks} src[i * block_stride + r] for r < num_rows (row-major bf16 src [.., rows, H],
        e.g. a gather over num_blocks chips) -> bf16 TILE [1, 1, num_rows, H]: the sum and the tilize in one pass.

        Args:
            src (ttnn.Tensor): [.., rows, H] bf16 ROW_MAJOR DRAM (H % 1024 == 0).
            num_rows (int): a multiple of 32.
            num_blocks (int): >= 2.
            block_stride (int): rows between blocks.

        Keyword Args:
            output (ttnn.Tensor, optional): preallocated output.
        )doc",
        &moe_ag_sum_rows_tiled,
        nb::arg("src").noconvert(),
        nb::arg("num_rows"),
        nb::arg("num_blocks"),
        nb::arg("block_stride"),
        nb::kw_only(),
        nb::arg("output") = nb::none());

    ttnn::bind_function<"moe_ag_add_rows", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        out[i] = a[a_offset + i] + b[b_offset + i] for i < num_rows (row-major bf16, width H) -> row-major
        [1, 1, num_rows, H]. info_offset: b_offset per device from the chip info (word 1) instead.
        )doc",
        &moe_ag_add_rows,
        nb::arg("a").noconvert(),
        nb::arg("b").noconvert(),
        nb::arg("chip_info").noconvert(),
        nb::arg("num_rows"),
        nb::kw_only(),
        nb::arg("a_offset") = 0,
        nb::arg("b_offset") = 0,
        nb::arg("info_offset") = false,
        nb::arg("output") = nb::none());

    ttnn::bind_function<"moe_ag_untilize_active", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        y bfp8 TILE [.., rows, H] -> row-major bf16 [rows, H], only the tile rows that hold tokens (each local
        expert's ceil(count / 32) tile rows at its region; counts / regions / the local-slot map read on device).
        )doc",
        &moe_ag_untilize_active,
        nb::arg("y").noconvert(),
        nb::arg("counts").noconvert(),
        nb::arg("regions").noconvert(),
        nb::arg("local_slot_map").noconvert(),
        nb::arg("experts_per_chip"),
        nb::kw_only(),
        nb::arg("tiles_per_block") = 32,
        nb::arg("output") = nb::none());

    ttnn::bind_function<"moe_ag_untilize_x", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        x bf16 TILE [.., S, H] -> row-major [1, 1, S H / 1024, 1024]: token row g in 2 KB pages (H / 1024) g ..,
        so the indexed expert's reads of a token spread over H / 1024 DRAM banks.
        )doc",
        &moe_ag_untilize_x,
        nb::arg("x").noconvert(),
        nb::kw_only(),
        nb::arg("output") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_ag::detail
