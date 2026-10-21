// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "entry_correction_nanobind.hpp"

#include "entry_correction.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace ttnn::operations::experimental::kda::entry_correction::detail {

void bind_entry_correction(nb::module_& mod) {
    ttnn::bind_function<"entry_correction", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Every chunk's output and this rank's final state from the summary's emitted terms
        (``summarize_chunk_recurrence(emit_outputs=True)``) once the entry states are known: the recurrence is
        affine in its seed, so ``output_c = F0_c + (FI_c - F0_c) @ entry`` and ``final = A @ entry + B``, all
        chunks in parallel instead of a second sequential pass. ``entry`` applies before the chronological reset
        chunk and ``tail_entry`` from it on. Each head's chunks are split over ``len(cores) / (B*H)`` cores, and
        ``zero_output``/``identity_output`` are height-sharded over ``cores`` so each core's chunk rows are in its L1.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: BFLOAT16 ``[B*H, chunks, 32, V]`` chunk outputs and the FP32
            ``[B*H, K, V]`` final state.
        )doc",
        &ttnn::experimental::kda::entry_correction,
        nb::arg("zero_output").noconvert(),
        nb::arg("identity_output").noconvert(),
        nb::arg("final_a").noconvert(),
        nb::arg("final_b").noconvert(),
        nb::arg("entry").noconvert(),
        nb::arg("tail_entry").noconvert(),
        nb::kw_only(),
        nb::arg("actual_start").noconvert(),
        nb::arg("local_rows"),
        nb::arg("cores"),
        nb::arg("actual_end") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::entry_correction::detail
