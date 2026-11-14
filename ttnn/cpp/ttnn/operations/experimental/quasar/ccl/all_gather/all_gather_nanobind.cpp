// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_nanobind.hpp"

#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "all_gather.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_all_gather(nb::module_& mod) {
    const auto* doc = R"doc(
        Intra-chip all-gather (Quasar-native, Metal 2.0).

        The input is a TILE tensor height- or width-sharded in L1 over N clusters, one shard per cluster;
        the cluster holding shard r is rank r and shard r is rank r's slice. Returns a tensor of shape
        [N, *input.shape], height-sharded over the same clusters, whose every shard is the full gathered
        tensor (``output[r] == input`` for all r).

        Args:
            input_tensor (ttnn.Tensor): height- or width-sharded TILE tensor.

        Keyword Args:
            topology (str): "direct" (one kernel of reader_threads DM threads per rank pulls every slice
                straight into the local shard, one NoC read per packet) or "ring" (unidirectional
                store-and-forward ring with per-thread ready credits). Default "direct".
            num_workers_per_rank (int): ring only, clusters per rank (the shard cluster plus helpers). Default 1.
            reader_threads (int): DM threads of the reader kernel, 1..6 for direct; ring needs
                reader_threads == writer_threads and at most 6 in total. Default 2.
            writer_threads (int): ring only, DM threads of the writer kernel. Default 2.
            tiles_per_packet (int): contiguous tiles per NoC transaction. Default 4.
            chunks_per_sync (int): ring only, packets per ready credit. Default 1.
    )doc";

    ttnn::bind_function<"all_gather", "ttnn.experimental.quasar.">(
        mod,
        doc,
        &ttnn::operations::experimental::quasar::all_gather,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("topology") = "direct",
        nb::arg("num_workers_per_rank") = 1,
        nb::arg("reader_threads") = 2,
        nb::arg("writer_threads") = 2,
        nb::arg("tiles_per_packet") = 4,
        nb::arg("chunks_per_sync") = 1);
}

}  // namespace ttnn::operations::experimental::quasar::detail
