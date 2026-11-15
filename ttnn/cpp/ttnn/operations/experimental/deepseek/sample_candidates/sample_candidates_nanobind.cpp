// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sample_candidates_nanobind.hpp"

#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/sample_candidates/sample_candidates.hpp"

namespace ttnn::operations::experimental::deepseek::detail {

void bind_sample_candidates(::nanobind::module_& mod) {
    namespace nb = nanobind;
    ttnn::bind_function<"decode_sample_candidates", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        The top-``k`` candidates of one row of logits, laid out as ``ttnn.sampling`` takes them, in one program.

        Args:
            logits (ttnn.Tensor): ``[1, 1, 1, V]`` BFLOAT16 TILE DRAM interleaved.
            inv_temp (ttnn.Tensor): ``[1, 1, 1, 1]`` BFLOAT16 TILE; the values are multiplied by it.
            index_offset (ttnn.Tensor): ``[1, 1, 1, 1]`` INT32 ROW_MAJOR; added to each index.
            k (int): candidates kept (default 32).
            fill_value (float): the value of the ``k`` padding candidates after them.
            num_workers (int): cores scanning the logits; core 0 merges their lists.

        Returns:
            List[ttnn.Tensor]: values ``[1, 1, 1, 2k]`` BFLOAT16 TILE (the ``k`` largest, descending, ties by lower
            index, times ``inv_temp``, then ``fill_value``) and indices ``[1, 1, 1, 2k]`` INT32 ROW_MAJOR (their
            columns plus ``index_offset``, then 0), DRAM interleaved.
        )doc",
        &ttnn::experimental::deepseek::decode_sample_candidates,
        nb::arg("logits"),
        nb::arg("inv_temp"),
        nb::arg("index_offset"),
        nb::arg("k") = 32,
        nb::arg("fill_value") = -1.0e9F,
        nb::arg("num_workers") = 64);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
