// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "exchange_histories_nanobind.hpp"

#include "exchange_histories.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

namespace ttnn::operations::experimental::kda::exchange_histories::detail {

void bind_exchange_histories(nb::module_& mod) {
    ttnn::bind_function<"exchange_histories", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Exchange the convolution histories along the sequence-parallel line over the fabric.

        Every rank sends its outgoing history (the three rows before the next physical rank's segment) to that
        rank, which receives it as its predecessor history, and the rank owning the chronologically last valid
        token sends its local final history to every rank as the replacement carry. The chronology is derived on
        device from ``actual_start`` and the optional ``actual_end``, as ``chronological_selections`` derives it, and
        both histories are gathered from the projection's rows that this chronology selects.

        Args:
            projected (ttnn.Tensor): BFLOAT16 tiled interleaved projection of ``local_rows`` rows whose leading
                ``width`` columns are the convolution channels.

        Keyword Args:
            width (int): The channel columns, a multiple of 32.
            actual_start (ttnn.Tensor): Replicated UINT32 row-major scalar, the chunk's first absolute position.
            local_rows (int): Positive, 32-aligned token rows per SP device.
            actual_end (ttnn.Tensor, optional): Replicated UINT32 row-major exclusive valid end.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory. Defaults to DRAM.
            sequence_parallel_axis (int, optional): Mesh axis partitioning the sequence.

        Returns:
            list[ttnn.Tensor]: BFLOAT16 row-major ``[1, 3, width]`` predecessor history and final history.
        )doc",
        &ttnn::experimental::kda::exchange_histories,
        nb::arg("projected").noconvert(),
        nb::kw_only(),
        nb::arg("width"),
        nb::arg("actual_start").noconvert(),
        nb::arg("local_rows"),
        nb::arg("actual_end") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0);
}

}  // namespace ttnn::operations::experimental::kda::exchange_histories::detail
