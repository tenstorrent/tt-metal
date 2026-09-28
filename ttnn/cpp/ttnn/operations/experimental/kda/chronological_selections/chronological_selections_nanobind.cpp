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

    ttnn::bind_function<"chronological_selections", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Derive KDA history and recurrent-state selection records on device.

        Args:
            actual_start (ttnn.Tensor): Replicated, interleaved UINT32 row-major
                device scalar containing the absolute position of the first token.
                Its value must be 32-aligned on every execution; runtime value
                validation is the caller's responsibility.
            sequence_parallel_axis (int): Mesh axis partitioning the sequence.
            local_rows (int): Positive, 32-aligned physical token count per rank.
            batch_heads (int): Positive number of batch-head pairs in the state.
            key_dim (int): Positive state key dimension.
            value_dim (int): Positive state value dimension.

        The scalar is read on every execution without host readback. Keep its
        address stable and update its contents before replaying a captured trace.
        Each device's rank comes from its mesh coordinate. The interval occupies
        the full physical capacity; this operation does not accept an end bound.

        Returns:
            ttnn.Tensor: Interleaved UINT32 row-major DRAM table of shape
                ``[7 + 2 * SP_size, 8]`` per device. This private representation
                contains three history-index records, paired start/exclusive-end
                bounds for local entry and final state, and one bounds pair per
                chronological affine-transform step. Consumers must use the shared
                ``_selection_layout`` definitions rather than hard-coded offsets.
        )doc",
        &ttnn::experimental::kda::chronological_selections,
        nb::arg("actual_start").noconvert(),
        nb::arg("sequence_parallel_axis"),
        nb::arg("local_rows"),
        nb::arg("batch_heads"),
        nb::arg("key_dim"),
        nb::arg("value_dim"));
}
}  // namespace ttnn::operations::experimental::kda::chronological_selections::detail
