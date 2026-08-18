// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_padding_config_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "moe_padding_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_padding_config::detail {

void bind_moe_padding_config(nb::module_& mod) {
    ttnn::bind_function<"moe_padding_config", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Build the per-device MoE padding config -- ``[local_real_tokens, pad_side]`` -- ON DEVICE
            from this chunk's ``actual_start`` / ``actual_end``.

            This is the traceable counterpart of the host builder
            (``TtMoEGatePrefill.build_padding_config``), which ends in a ``ttnn.from_torch`` and so
            cannot run inside a trace capture. The consumers (the ``moe_grouped_topk`` gate and the
            ``dispatch`` op) already read ``padding_config`` as a device tensor, so moving only its
            producer on-device makes the whole padding-aware path traceable with no downstream change:
            one captured program recomputes the correct config on every replay.

            Counts are derived for the KV-pad-aware ROTATED block-cyclic layout, mirroring
            ``update_padded_kv_cache``'s writer, and reduce exactly to the sequential formula when
            ``actual_start`` is slab-aligned (including 0) -- so one op covers the rotated and
            non-rotated cases alike.

            NOT covered: the ``is_balanced=True`` zigzag placement, whose per-device count is not
            expressible in this form. Rotated chunked prefill implies ``is_balanced=False``, so that
            combination is unreachable here; balanced callers must keep using the host builder.

            In place: returns a handle to ``config``.

            Args:
                config (ttnn.Tensor): per-device [.., 2] UINT32 ROW_MAJOR DRAM tensor written in
                    place. The global tensor is [sp_factor, 2] sharded along ``cluster_axis``, so each
                    chip holds one row. Caller-owned; must be PERSISTENT (a stable address) for a
                    captured trace to keep writing the same buffer across replays.
                actual_start (ttnn.Tensor): 1-element uint32 DRAM tensor ([1,1,1,1], ROW_MAJOR,
                    replicated across the mesh) holding the absolute KV position of this chunk's
                    first real token. Read on-device (element [0]).
                actual_end (ttnn.Tensor): 1-element uint32 DRAM tensor (same layout) holding one past
                    this chunk's last real token. Read on-device (element [0]).
                tokens_per_chip (int): tokens this chip carries per chunk (the MoE's sp_dim).
                    Structural.
                pad_side (int): 0 = right, 1 = left. Structural.
                cluster_axis (int): mesh axis the config is sharded along (the SP axis, 0 or 1).
                    Structural.

            Returns:
                ttnn.Tensor: handle to ``config``, with this chip's row written.
        )doc",
        &moe_padding_config,
        nb::arg("config").noconvert(),
        nb::arg("actual_start").noconvert(),
        nb::arg("actual_end").noconvert(),
        nb::arg("tokens_per_chip"),
        nb::arg("pad_side"),
        nb::arg("cluster_axis"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_padding_config::detail
