// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_direct_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_direct/reduce_scatter_minimal_direct.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_reduce_scatter_minimal_direct(nb::module_& mod) {
    ttnn::bind_function<"reduce_scatter_minimal_direct", "ttnn.experimental.">(
        mod,
        R"doc(
        Experimental direct (one-shot) reduce-scatter across a 1D ring of devices.

        Every device unicasts each destination's slice straight to that destination over the fabric (a
        multi-hop unicast; no intermediate device stages or accumulates it), then reduces the arrivals
        with its own slice. One fabric traversal instead of the ring's N/2 store-and-forward steps, at
        ~2.3x the link traffic: a latency play for small/medium shapes.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor, TILE layout, any rank >= 2.
            dim (int): Dimension to scatter. Any dim, provided it splits into `num_devices` whole
                slices in tile/page units (the two innermost dims are counted in tiles).

        Keyword Args:
            cluster_axis (int, optional): mesh axis to run the collective on. Defaults to the active axis.
            num_links (int, optional): fabric links to spread the collective over; one worker core per
                link, each owning that link's forward and backward connection plus a contiguous
                sub-range of every slice's chunks. Defaults to every link available on the active axis.
            memory_config (ttnn.MemoryConfig, optional): output memory config. Defaults to input's.
            persistent_buffers (List[ttnn.Tensor], optional): {output, staging} from
                reduce_scatter_minimal_direct_create_persistent_buffers, reused to skip re-allocation.
            subdevice_id (ttnn.SubDeviceId, optional).
            sub_core_grid (ttnn.CoreRangeSet, optional).

        Returns:
            ttnn.Tensor: the reduced output slice for this device.
        )doc",
        &ttnn::experimental::reduce_scatter_minimal_direct,
        nb::arg("input_tensor"),
        nb::arg("dim"),
        nb::kw_only(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("persistent_buffers") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("sub_core_grid") = nb::none());

    ttnn::bind_function<"reduce_scatter_minimal_direct_create_persistent_buffers", "ttnn.experimental.">(
        mod,
        R"doc(
        Allocate the persistent buffer set {output, staging} sized to a given input, for reuse across
        reduce_scatter_minimal_direct invocations. `dim` and `cluster_axis` must match those passed to
        the op.

        Returns:
            List[ttnn.Tensor]: [output, staging] on the input tensor's device.
        )doc",
        &ttnn::experimental::reduce_scatter_minimal_direct_create_persistent_buffers,
        nb::arg("input_tensor"),
        nb::arg("dim"),
        nb::kw_only(),
        nb::arg("cluster_axis") = nb::none());
}

}  // namespace ttnn::operations::experimental::ccl
