// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "block_keys_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "block_keys.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::block_keys {

// The op refuses only through TT_FATAL (RuntimeError), so no support-refusal translator is registered.
void bind_block_keys_operation(nb::module_& mod) {
    ttnn::bind_function<"block_keys">(
        mod,
        R"doc(
        Key ids of the positions a top-k picked in block_gather's row: out[.., r, i] = ids[.., r, pos / block] * block
        + pos % block, 0xFFFFFFFF where pos is none (>= K * block, e.g. 0xFFFFFFFF) or its block id is >= width / block.

        Args:
            pos (ttnn.Tensor): [.., R, KP] uint32 ROW_MAJOR, DRAM interleaved: positions in block_gather's [.., R, K block] row.
            ids (ttnn.Tensor): [.., R, K] uint32 ROW_MAJOR, DRAM interleaved: the block ids block_gather read.

        Keyword Args:
            block (int): the block size B, a power of two <= 32.
            width (int): the scores' width W; block ids >= W / B are none.
            output_tensor (ttnn.Tensor, optional): preallocated [.., R, KP] uint32 ROW_MAJOR DRAM output, returned as is.
            memory_config (ttnn.MemoryConfig, optional): output placement; default DRAM_MEMORY_CONFIG (DRAM interleaved only).
            max_grid (ttnn.CoreCoord, optional): caps the worker grid the rows are split over.

        Returns:
            ttnn.Tensor: [.., R, KP] uint32 ROW_MAJOR key ids.

        Raises:
            RuntimeError: an input outside the contract above.
        )doc",
        &ttnn::block_keys,
        nb::arg("pos"),
        nb::arg("ids"),
        nb::kw_only(),
        nb::arg("block"),
        nb::arg("width"),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("max_grid") = nb::none());
}

}  // namespace ttnn::operations::block_keys
