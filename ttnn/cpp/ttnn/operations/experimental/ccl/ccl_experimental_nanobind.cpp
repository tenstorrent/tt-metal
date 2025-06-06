// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/operations/experimental/ccl/ccl_experimental_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/ccl/all_gather_matmul/all_gather_matmul_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/rms_allgather/rms_allgather_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce/all_reduce_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/all_gather_concat_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_async/reduce_scatter_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter/llama_reduce_scatter_nanobind.hpp"

namespace ttnn::operations::experimental::ccl {

void py_module(nb::module_& mod) {
    ccl::bind_fused_rms_1_1_32_8192(mod);
    ccl::bind_all_gather_matmul(mod);
    ccl::bind_all_reduce(mod);
    ccl::bind_all_gather_async(mod);
    ccl::bind_all_gather_concat(mod);
    ccl::bind_reduce_scatter_async(mod);
    ccl::bind_all_reduce_async(mod);
    ccl::bind_llama_reduce_scatter(mod);
}

}  // namespace ttnn::operations::experimental::ccl
