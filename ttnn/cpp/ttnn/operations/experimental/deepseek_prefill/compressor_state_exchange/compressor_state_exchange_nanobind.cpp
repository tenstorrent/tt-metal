// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compressor_state_exchange_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "compressor_state_exchange.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::compressor_state_exchange::detail {

void bind_compressor_state_exchange(nb::module_& mod) {
    ttnn::bind_function<"compressor_state_exchange", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Shift Blaze-compatible compressor state to the next SP rank.

            Rank zero receives the injected temporal state. Every other rank receives the
            complete 64-row KV and score state from its predecessor along ``cluster_axis``.
            Mesh coordinates on the other axis are independent TP lanes.

            All inputs are BFLOAT16 TILE tensors in interleaved memory. Each local mesh shard
            has shape ``[B, 1, 64, 512]``. The initial states have the same distributed tensor
            spec as the local states and contain the prior-call state in every SP shard; only
            rank zero's copy is preserved in the result.

            The local input tensors are not modified. They remain the outgoing compressor
            states, and the final active SP rank's local state is directly compatible with
            Blaze decode migration.

            Returns:
                Tuple of predecessor KV state and predecessor score state.
        )doc",
        &compressor_state_exchange,
        nb::arg("local_kv_state").noconvert(),
        nb::arg("local_score_state").noconvert(),
        nb::arg("initial_kv_state").noconvert(),
        nb::arg("initial_score_state").noconvert(),
        nb::kw_only(),
        nb::arg("cluster_axis") = 0,
        nb::arg("topology").noconvert() = ::ttnn::ccl::Topology::Linear);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::compressor_state_exchange::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_compressor_state_exchange(::nanobind::module_& mod) {
    compressor_state_exchange::detail::bind_compressor_state_exchange(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
