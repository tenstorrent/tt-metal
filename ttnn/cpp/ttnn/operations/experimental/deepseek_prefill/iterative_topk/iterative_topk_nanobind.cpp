// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "iterative_topk_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/iterative_topk/iterative_topk.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk::detail {

void bind_iterative_topk(nb::module_& mod) {
    ttnn::bind_function<"iterative_topk", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Computes top-k values and indices by iteratively finding the maximum value k times,
            masking each found maximum with -infinity before the next iteration. Operates on the
            last dimension of the input tensor.

            Args:
                input (ttnn.Tensor): Input tensor (dtype must be FLOAT32, layout must be ROW_MAJOR, memory must be interleaved). The last dimension is the dimension over which top-k is computed.
                k (int): Number of top elements to find.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensors. Defaults to None, which uses the input tensor's memory config.

            Returns:
                Tuple[ttnn.Tensor, ttnn.Tensor]: A tuple containing the top-k values (dtype FLOAT32) and their indices (dtype UINT32). Both outputs have the same shape as the input except the last dimension is k.
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::iterative_topk::iterative_topk,
        nb::arg("input").noconvert(),
        nb::kw_only(),
        nb::arg("k"),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk::detail
