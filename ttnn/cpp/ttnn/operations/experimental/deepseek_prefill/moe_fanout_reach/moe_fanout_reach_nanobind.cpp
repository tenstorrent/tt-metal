// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_fanout_reach_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <tt-metalium/sub_device_types.hpp>

#include "ttnn-nanobind/bind_function.hpp"
#include "moe_fanout_reach.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach::detail {
void bind_experimental_moe_fanout_reach_operation(nb::module_& mod) {
    ttnn::bind_function<"moe_fanout_reach", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        This device's row of the multicast reach table `dispatch_fabric2d(fanout=True)` sizes its
        chunks from.

        Under multicast one copy per (token, direction) crosses a cable: it travels the ring and every
        chip en route keeps the pages addressed to it. A chunk is therefore (origin, hop), and its
        length is "how many of that origin's tokens are still travelling this way at hop h" -- which no
        per-expert count can express, because per-expert counts are marginals. reach[direction][hop] is
        exactly that number.

        The table is POST-DROP, and that is why it cannot be folded into `masked_bincount` or derived
        inside `dispatch_fabric2d`. A pick is dropped when its per-expert allocator has already reached
        the destination buffer's capacity, and the allocator is seeded from `global_dispatch_offsets`,
        which `offset_cumsum` computes FROM the histograms -- so the drop rule is not knowable when the
        histograms are built. And a relay sizes chunks from OTHER origins' reach, which depends on
        those origins' indices; indices are sharded per chip, so a gather is unavoidable.

        A reach table that OVERSTATES deadlocks the axis: the origin sends fewer pages than the relay
        waits for. Exactness, not closeness, is the requirement, so this op replays the same allocator
        and the same drop rule the transport replays.

        Args:
            * :attr:`indices_tensor`: top-k expert ids per token, UINT16 ROW_MAJOR of shape
              [.., seq_len, num_experts_per_tok] -- the SAME tensor `dispatch_fabric2d` is handed.
              Sharing one tensor is the point: reach describes what that routing sends, and a second
              copy is a second chance for the two to disagree.
            * :attr:`expert_dispatch_table`: INT32 ROW_MAJOR, global expert id -> row in the dispatch
              group, -1 when the expert is not in this group. A trailing sentinel column is allowed and
              ignored; an expert id at or above num_routed_experts is refused, which is the answer the
              sentinel's -1 would have given.
            * :attr:`global_dispatch_offsets`: THIS device's row of the offsets table -- INT32 or
              UINT32 ROW_MAJOR with exactly num_routed_experts elements, i.e. `offset_cumsum`'s
              `global_dispatch_offsets`, not its all-rows table. It seeds the per-expert allocator.
            * :attr:`num_routed_experts`, :attr:`num_experts_per_tok`: the routing shape.
            * :attr:`dispatch_group_size`: the ring extent, which must equal the mesh extent on
              cluster_axis.
            * :attr:`max_dispatch_buffer_token_size`: the destination buffer's shared token capacity,
              the same value `dispatch_fabric2d` is given. A pick past it is dropped while its
              per-expert counter still advances.
            * :attr:`cluster_axis`: the axis the transport relays over.
            * :attr:`subdevice_id`: which sub-device's cores the op may use; defaults to the first.

        Returns:
            INT32 ROW_MAJOR [1, 2, cluster_axis_extent / 2 + 2], per device. Index 0 is the clockwise
            direction. Hop 0 is unused and zero; the entry at extent / 2 + 1 is a terminating zero, so
            that reach[h] - reach[h + 1] is the number of tokens whose farthest hop is exactly h for
            every h including extent / 2.

            All-gather this along cluster_axis to get the [extent, 2, extent / 2 + 2] table
            `dispatch_fabric2d` validates: a relay sizes a chunk it neither wrote nor receives, so it
            needs every origin's row.

        Note:
            Reach is computed over every token in the sequence. That agrees with a
            `dispatch_fabric2d` call carrying a `padding_config`, which walks only the real tokens,
            because supplying that config asserts the padded ones are sentinel-marked and so contribute
            no page here either.

        )doc",
        &moe_fanout_reach,
        nb::arg("indices_tensor").noconvert(),
        nb::arg("expert_dispatch_table").noconvert(),
        nb::arg("global_dispatch_offsets").noconvert(),
        nb::kw_only(),
        nb::arg("num_routed_experts"),
        nb::arg("num_experts_per_tok"),
        nb::arg("dispatch_group_size"),
        nb::arg("max_dispatch_buffer_token_size"),
        nb::arg("cluster_axis"),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("subdevice_id") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach::detail
