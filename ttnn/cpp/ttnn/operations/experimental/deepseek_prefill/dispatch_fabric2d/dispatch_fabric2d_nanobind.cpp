// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/sub_device_types.hpp>

#include "ttnn-nanobind/bind_function.hpp"
#include "dispatch_fabric2d.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d::detail {
void bind_experimental_dispatch_fabric2d_operation(nb::module_& mod) {
    ttnn::bind_function<"dispatch_fabric2d", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        MoE prefill dispatch over FABRIC_2D. Each token is sent to the chips that host its top-k experts,
        one hop at a time around the ring of chips on `cluster_axis`. Each chip on the way stores the token
        in a DRAM forwarding buffer and forwards it to the next chip.

        Inputs. All are interleaved, and all except input_tensor are ROW_MAJOR:

            input_tensor          tokens, BFLOAT16, ROW_MAJOR (one token per page) or TILE. A TILE input is
                                  untilized on device and needs emb_dim to be a multiple of 32.
            indices_tensor        top-k expert ids, UINT16, (..., seq_len_per_chip, num_experts_per_tok).
            expert_offsets        INT32 or UINT32, (..., extent, num_routed_experts): where each source
                                  chip's tokens start in each expert's region, one row per source chip.
                                  Pass offset_cumsum's all_global_dispatch_offsets, replicated along
                                  cluster_axis.
            expert_dispatch_table INT32, num_routed_experts + 1 columns: global expert id -> chip in the
                                  dispatch group, or -1 when the expert is not in this group. The extra
                                  last column must be -1; padded tokens look it up.
            expert_token_counts   INT32 or UINT32, (..., num_routed_experts): tokens per expert, summed over
                                  all source chips.
            expert_region_offsets INT32 or UINT32, (..., num_routed_experts): where each expert's region
                                  starts in the output buffer.
            padding_config        optional INT32 or UINT32 [real_token_count, pad_side]. With right padding
                                  (pad_side 0) only the first real_token_count tokens are routed; other
                                  sides are ignored. Padded tokens must resolve to no expert.
            subdevice_id          sub-device whose Tensix cores the op may use. Defaults to the first
                                  sub-device, which is the whole compute grid when no sub-device manager
                                  is loaded.

        Returns [dispatched_buffer, metadata], both per device and ROW_MAJOR, in `memory_config`.
        dispatched_buffer is (1, 1, max_dispatch_buffer_token_size, emb_dim) BFLOAT16. metadata is
        (1, 1, max_dispatch_buffer_token_size, 3) INT32 and holds (source chip, token index, topk index) at
        the same page as its token. A token past the buffer capacity is dropped but still counts toward its
        expert's offsets.

        Checked constraints:

            topology              Ring or Torus, and cluster_axis must be wrap-wired.
            cluster_axis extent   even and at least 4.
            num_links             1 to 4, and the axis must have that many forwarding links.
            num_routed_experts    a multiple of 16.
            metadata_len          3.
            memory_config         interleaved.
            subdevice_id          must contain the worker core nearest each ethernet core the op sends on,
                                  one per link direction (2 * num_links cores). A TILE input also needs at
                                  least one core in the row under those.

        Not checked: every tensor must be in DRAM, and expert_offsets must be replicated along
        cluster_axis. fp8 input and output are not supported. cluster_axis other than 0 is untested.
        )doc",
        &dispatch_fabric2d,
        nb::arg("input_tensor"),
        nb::arg("indices_tensor"),
        nb::arg("expert_offsets"),
        nb::arg("expert_dispatch_table"),
        nb::arg("expert_token_counts"),
        nb::arg("expert_region_offsets"),
        nb::arg("padding_config") = std::nullopt,
        nb::arg("experts_per_chip"),
        nb::arg("num_routed_experts"),
        nb::arg("num_experts_per_tok"),
        nb::arg("metadata_len"),
        nb::arg("max_dispatch_buffer_token_size"),
        nb::arg("seq_len_per_chip"),
        nb::arg("cluster_axis") = 0,
        nb::arg("num_links") = 1,
        nb::arg("topology"),
        nb::arg("memory_config"),
        nb::arg("subdevice_id") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d::detail
