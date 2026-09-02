// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "broadcast_ring_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "broadcast_ring.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::ccl {

void bind_broadcast_ring(nb::module_& mod) {
    const auto* doc =
        R"doc(
        One-sender ring broadcast (experimental): broadcast the shard at ``sender_ring_index`` (along
        ``cluster_axis``) to every device on that ring line, via manual per-hop unicast relay over
        FABRIC_1D / FABRIC_1D_RING. Runs independently on each line parallel to the orthogonal axis, so a
        tp-sharded input broadcasts each orthogonal row's own data (unlike ttnn.broadcast, which requires
        the orthogonal axis replicated and FABRIC_2D).

        v1: single sender; one-way around the ring (requires the wrap link -> Ring topology).

        Args:
            input_tensor (ttnn.Tensor): input, sharded on ``cluster_axis`` (and optionally the other axis).
            sender_ring_index (int): index along ``cluster_axis`` whose shard is broadcast.
            cluster_axis (int): the ring axis to broadcast along.

        Keyword Args:
            num_links (int, optional): number of fabric links. Defaults to auto.
            memory_config (ttnn.MemoryConfig, optional): output memory config. Defaults to the input's.
            topology (ttnn.Topology, optional): must be Ring for v1. Defaults to Ring.
            subdevice_id (ttnn.SubDeviceId, optional): worker sub-device id.

        Returns:
            ttnn.Tensor: same shape as the input; every device on the ring line holds the sender shard's data.
        )doc";

    ttnn::bind_function<"broadcast_ring", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::broadcast_ring,
        nb::arg("input_tensor"),
        nb::arg("sender_ring_index"),
        nb::arg("cluster_axis"),
        nb::kw_only(),
        nb::arg("num_links") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("topology").noconvert() = ttnn::ccl::Topology::Ring,
        nb::arg("subdevice_id") = nb::none());
}

}  // namespace ttnn::operations::experimental::ccl
