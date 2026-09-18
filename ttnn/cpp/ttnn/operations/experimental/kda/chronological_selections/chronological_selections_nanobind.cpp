// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chronological_selections_nanobind.hpp"
#include "chronological_selections.hpp"
#include "device/kernels/chronology.hpp"
#include "ttnn-nanobind/bind_function.hpp"
namespace ttnn::operations::experimental::kda::chronological_selections::detail {
void bind_chronological_selections(nb::module_& mod) {
    // Private representation contract shared with the model's ChronologicalSelections adapter.
    auto layout = mod.def_submodule("_selection_layout");
    using namespace kda_chronology::selection;
    layout.attr("HISTORY_ROWS") = history_rows;
    layout.attr("SLICE_RANK") = slice_rank;
    layout.attr("OUTGOING_HISTORY") = outgoing_history;
    layout.attr("PREDECESSOR_HISTORY") = predecessor_history;
    layout.attr("FINAL_HISTORY") = final_history;
    layout.attr("LOCAL_ENTRY_STATE") = local_entry_state;
    layout.attr("FINAL_STATE") = final_state;
    layout.def("affine_transform", &affine_transform);
    layout.def("local_final_history", &local_final_history);

    ttnn::bind_function<"chronological_selections", "ttnn.experimental.kda.">(
        mod,
        "Derive KDA selection records from device actual_start and static mesh geometry.",
        &ttnn::experimental::kda::chronological_selections,
        nb::arg("actual_start").noconvert(),
        nb::arg("sequence_parallel_axis"),
        nb::arg("local_rows"),
        nb::arg("batch_heads"),
        nb::arg("key_dim"),
        nb::arg("value_dim"),
        nb::kw_only(),
        nb::arg("actual_end") = nb::none());
}
}  // namespace ttnn::operations::experimental::kda::chronological_selections::detail
