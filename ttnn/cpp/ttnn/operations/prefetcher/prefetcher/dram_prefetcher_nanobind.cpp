// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include "ttnn-nanobind/bind_function.hpp"
#include "dram_prefetcher.hpp"

namespace ttnn::operations::dram_prefetcher::detail {

void bind_dram_prefetcher_operation(nb::module_& mod) {
    ttnn::bind_function<"dram_prefetcher">(
        mod,
        R"doc(
            Asynchronously pre-fetch tensors from DRAM into the neighbouring L1 cores.
            Reader cores, one per DRAM bank, push each weight block to consumer cores through either a
            global circular buffer or PrefetcherPipes.

            Args:
                tensors (List[ttnn.Tensor]): The tensors to pre-fetch, followed by the address tensor: a row-major UINT32 tensor, height sharded in L1 over the reader cores, that holds the tensors' DRAM addresses as [t1_l1, t2_l1, ..., t1_l2, t2_l2, ..., t1_l3, t2_l3, ...].
                num_layers (int): The number of layers in the pipeline or the model for which tensors need to be pre-fetched.
                global_cb (GlobalCircularBuffer, optional): A worker-sender global circular buffer, used internally to manage data movement across dram reader cores, and downstream consumer cores. Defaults to None; exactly one of `global_cb` and `prefetcher_pipes` must be given.

            Keyword Args:
                enable_performance_mode (bool, optional): If set to true, the operation will be optimized for performance. May lead to ND behavior on wormhole 4U systems! On the `prefetcher_pipes` path it affects only the DRAM reads.
                prefetcher_pipes (List[ttnn.experimental.PrefetcherPipe], optional): Worker-sender PrefetcherPipes to deliver into instead of `global_cb`, one per reader core (its sender). In the order given, as `global_cb` takes its senders, the i-th pipe's sender reads DRAM bank i; pipes beyond the reader count are not used. Every reader's pipe needs the same number of receivers. Each receiver gets one block slice per pipe entry, so the entry size follows the tensor being delivered and must fit the ring. The delivery contract is in tt_metal/impl/buffers/prefetcher_matmul_design.md, "Delivering into PrefetcherPipes". Keep the pipes alive for as long as the program cache may hold a program built against them. Defaults to an empty list (none).

            Returns:
                ttnn.Tensor: empty tensor (TODO: Should return None)
        )doc",
        &ttnn::dram_prefetcher,
        nb::arg("tensors"),
        nb::arg("num_layers"),
        nb::arg("global_cb") = nb::none(),
        nb::kw_only(),
        nb::arg("enable_performance_mode") = false,
        nb::arg("prefetcher_pipes") = ttnn::PrefetcherPipeList{});
}

void bind_dram_prefetcher(nb::module_& mod) { bind_dram_prefetcher_operation(mod); }

}  // namespace ttnn::operations::dram_prefetcher::detail
