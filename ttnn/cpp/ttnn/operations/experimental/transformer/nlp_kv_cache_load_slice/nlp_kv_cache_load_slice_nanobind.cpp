// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_kv_cache_load_slice_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "nlp_kv_cache_load_slice.hpp"
#include "ttnn/types.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::transformer::detail {

void bind_nlp_kv_cache_load_slice(nb::module_& mod) {
    const auto* doc = R"doc(
            Unpad TT INTERLEAVED, TILE layout Tensor into a height sharded tensor. Typically used to unpad the KV cache from [B,n_heads,max_seq_length,head_dim] (or [n_heads,B,max_seq_length,head_dim]) into [B,n_heads,S,head_dim] (or [n_heads,B,S,head_dim]), where S = seq_len_end-seq_len_start. seq_len_start and seq_len_end are the start and end of the sequence length to unpad, and must be multiples of 32.
            Returns an output tensor that is height sharded on B x n_heads corees (note the B and n_heads dims are interchangeable), where each shard is [S, head_dim].
        )doc";

    mod.def(
        "nlp_kv_cache_load_slice",
        &ttnn::experimental::nlp_kv_cache_load_slice,
        doc,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("seq_len_start").noconvert(),
        nb::arg("seq_len_end").noconvert(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer::detail
