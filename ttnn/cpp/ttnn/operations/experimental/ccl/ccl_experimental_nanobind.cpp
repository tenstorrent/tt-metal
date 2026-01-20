// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ccl_experimental_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/ccl/rms_allgather/rms_allgather_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/all_gather_matmul_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/strided_all_gather_minimal_matmul_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/llama_all_gather_matmul_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/all_to_all_async/all_to_all_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/all_to_all_async_generic/all_to_all_async_generic_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/all_gather_concat_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/rs_matmul_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter/llama_reduce_scatter_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/llama_reduce_scatter_create_heads_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/ring_attention_all_gather_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_async/neighbor_pad_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/slice_reshard_async/slice_reshard_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/strided_all_gather_async_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/deepseek_minimal_broadcast_nanobind.hpp"

namespace ttnn::operations::experimental::ccl {

void py_module(nb::module_& mod) {
    ccl::bind_fused_rms_minimal(mod);
    ccl::bind_all_gather_matmul_async(mod);
    ccl::bind_strided_all_gather_minimal_matmul_async(mod);
    ccl::bind_llama_all_gather_matmul_async(mod);
    ccl::bind_all_gather_async(mod);
    ccl::bind_strided_all_gather_async(mod);
    ccl::bind_all_to_all_async(mod);
    ccl::bind_all_to_all_async_generic(mod);
    ccl::bind_all_gather_concat(mod);
    ccl::bind_matmul_reduce_scatter_async(mod);
    ccl::bind_rs_matmul(mod);
    ccl::bind_reduce_scatter_minimal_async(mod);
    ccl::bind_all_reduce_async(mod);
    ccl::bind_llama_reduce_scatter(mod);
    ccl::bind_llama_rs_create_heads(mod);
    ccl::bind_ring_attention_all_gather_async(mod);
    ccl::bind_send_async(mod);
    ccl::bind_recv_async(mod);
    ccl::bind_neighbor_pad_async(mod);
    ccl::bind_slice_reshard_async(mod);
    ccl::bind_deepseek_minimal_broadcast(mod);
}

}  // namespace ttnn::operations::experimental::ccl
