// SPDX-License-Identifier: Apache-2.0
#include "deltanet_prefill_chunked_nanobind.hpp"
#include <cstdint>
#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deltanet/deltanet_prefill_chunked.hpp"
namespace ttnn::operations::experimental::deltanet::detail {
void bind_deltanet_prefill_chunked(nb::module_& mod) {
    const auto* doc = "Chunked-parallel gated delta-rule prefill (recurrence only; decay scalings folded host-side).";
    ttnn::bind_function<"deltanet_prefill_chunked", "ttnn.experimental.">(
        mod, doc, &ttnn::experimental::deltanet_prefill_chunked,
        nb::arg("k"), nb::arg("q"), nb::arg("v"), nb::arg("z"),
        nb::arg("Kdec"), nb::arg("KiT"), nb::arg("Qd"),
        nb::arg("dcol"), nb::arg("betacol"), nb::arg("dlast"),
        nb::arg("recurrent_state"), nb::arg("norm_weight"),
        nb::kw_only(),
        nb::arg("num_heads"), nb::arg("k_head_dim"), nb::arg("v_head_dim"),
        nb::arg("chunk"), nb::arg("n_chunks"), nb::arg("seq_len"),
        nb::arg("memory_config") = nb::none());
}
}  // namespace ttnn::operations::experimental::deltanet::detail
