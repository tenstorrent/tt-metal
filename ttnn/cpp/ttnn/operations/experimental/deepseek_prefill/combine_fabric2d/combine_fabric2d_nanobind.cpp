// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "combine_fabric2d.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::detail {
void bind_experimental_combine_fabric2d_operation(nb::module_& mod) {
    ttnn::bind_function<"combine_fabric2d", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Isolated FABRIC_2D transfer experiment op. For each fabric eth core (`num_links` toward
        each neighbor along mesh `axis`), one worker core in that eth core's physical column runs a
        producer on the writer RISC and a receiver on the reader RISC. Every link is full duplex:
        the producer sends `num_tokens` chunks of `chunk_size_bytes` to the peer worker across the
        cable while the receiver consumes the peer's chunks into a `num_slots`-deep L1 ring, credited
        back through the producer's connection. No input tensors; returns a dummy tensor. Used to
        profile the fabric leg in isolation (inspect Tracy zones).
        )doc",
        &combine_fabric2d,
        nb::arg("device"),
        nb::arg("num_links") = 2,
        nb::arg("num_tokens") = 100,
        nb::arg("chunk_size_bytes") = 14336,
        nb::arg("num_slots") = 32,
        nb::arg("axis") = 0,
        nb::arg("topology") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::detail
