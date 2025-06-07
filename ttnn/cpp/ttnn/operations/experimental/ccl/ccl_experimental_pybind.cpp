// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ccl_experimental_pybind.hpp"

#include <pybind11/pybind11.h>

#include "ttnn/operations/experimental/ccl/all_gather_matmul/all_gather_matmul_pybind.hpp"
#include "ttnn/operations/experimental/ccl/rms_allgather/rms_allgather_pybind.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/all_gather_matmul_async_pybind.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async_pybind.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce/all_reduce_pybind.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async_pybind.hpp"
#include "ttnn/operations/experimental/ccl/all_broadcast_async/all_broadcast_async_pybind.hpp"
#include "ttnn/operations/experimental/ccl/all_to_all_async/all_to_all_async_pybind.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/all_gather_concat_pybind.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_async/reduce_scatter_pybind.hpp"
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/rs_matmul_pybind.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async_pybind.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async_pybind.hpp"
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter/llama_reduce_scatter_pybind.hpp"
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/llama_reduce_scatter_create_heads_pybind.hpp"

namespace ttnn::operations::experimental::ccl {

void py_module(pybind11::module& module) {
    ccl::bind_fused_rms_1_1_32_8192(module);
    ccl::py_bind_all_gather_matmul(module);
    ccl::py_bind_all_gather_matmul_async(module);
    ccl::py_bind_all_reduce(module);
    ccl::py_bind_all_gather_async(module);
    ccl::py_bind_all_broadcast_async(module);
    ccl::py_bind_all_to_all_async(module);
    ccl::py_bind_all_gather_concat(module);
    ccl::py_bind_matmul_reduce_scatter_async(module);
    ccl::py_bind_reduce_scatter_async(module);
    ccl::py_bind_rs_matmul(module);
    ccl::py_bind_reduce_scatter_minimal_async(module);
    ccl::py_bind_all_reduce_async(module);
    ccl::py_bind_llama_reduce_scatter(module);
    ccl::py_bind_llama_rs_create_heads(module);
}

}  // namespace ttnn::operations::experimental::ccl
