// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "block_gather_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "block_gather.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::block_gather {

namespace {

// A caller's output_tensor is returned as is (the Python op returned the very object it was given): it arrives as the
// pointer to the C++ tensor its Python object holds, and returning that pointer by reference makes nanobind hand back
// the existing Python object instead of wrapping a copy.
Tensor* block_gather_into(
    const Tensor& scores,
    const Tensor& ids,
    int64_t block,
    Tensor* output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::CoreCoord>& max_grid) {
    ttnn::block_gather(scores, ids, block, *output_tensor, memory_config, max_grid);
    return output_tensor;
}

}  // namespace

// The Python op refused only through asserts and declares no _op_contract refusals, so there is no exception
// translator: every refusal is a TT_FATAL (RuntimeError).
void bind_block_gather_operation(nb::module_& mod) {
    ttnn::bind_function<"block_gather">(
        mod,
        R"doc(
        Gathers, per row, the K blocks of ``block`` adjacent scores that ``ids`` names.

        scores [.., R, W] bf16 ROW_MAJOR + block ids [.., R, K] uint32 ROW_MAJOR (equal leading dims, DRAM
        interleaved) -> [.., R, K block] bf16 ROW_MAJOR: row r holds scores[r, ids[r, j] block + o] at j block + o
        (an id >= W / block, 0xFFFFFFFF included: -inf).

        Args:
            scores (ttnn.Tensor): [.., R, W] bfloat16 ROW_MAJOR, DRAM interleaved; W a multiple of 32 and of block.
            ids (ttnn.Tensor): [.., R, K] uint32 ROW_MAJOR, DRAM interleaved.

        Keyword Args:
            block (int): B, a power of two <= 32 dividing W.
            output_tensor (ttnn.Tensor, optional): preallocated [.., R, K block] bfloat16 ROW_MAJOR DRAM interleaved
                output, returned as is.
            memory_config (ttnn.MemoryConfig, optional): output placement; default DRAM interleaved.
            max_grid (ttnn.CoreCoord, optional): bounds the cores' rectangle; default the whole worker grid.

        Returns:
            ttnn.Tensor: [.., R, K block] bfloat16 ROW_MAJOR.

        Raises:
            RuntimeError: an input outside the contract above.
        )doc",
        // tried first: a caller's output_tensor (None does not match a pointer argument without .none())
        ttnn::overload_t(
            &block_gather_into,
            nb::arg("scores"),
            nb::arg("ids"),
            nb::kw_only(),
            nb::arg("block"),
            nb::arg("output_tensor"),
            nb::arg("memory_config") = nb::none(),
            nb::arg("max_grid") = nb::none(),
            nb::rv_policy::reference),
        ttnn::overload_t(
            &ttnn::block_gather,
            nb::arg("scores"),
            nb::arg("ids"),
            nb::kw_only(),
            nb::arg("block"),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("max_grid") = nb::none()));
}

}  // namespace ttnn::operations::block_gather
