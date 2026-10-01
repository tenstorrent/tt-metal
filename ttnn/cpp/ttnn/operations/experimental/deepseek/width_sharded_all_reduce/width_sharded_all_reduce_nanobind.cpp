// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "width_sharded_all_reduce_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/width_sharded_all_reduce/width_sharded_all_reduce.hpp"

namespace ttnn::operations::experimental::deepseek::width_sharded_all_reduce::detail {

void bind_width_sharded_all_reduce(nb::module_& mod) {
    const auto* doc =
        R"doc(
        All-reduce a ROW_MAJOR WIDTH_SHARDED tensor. Output layout and memory config match the input
        (same shard spec). A fused device program line-mcasts each 1x32 RM face and reduces on
        the tensor cores.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): ROW_MAJOR WIDTH_SHARDED device tensor. Last dim
              must be a multiple of 32.

        Keyword Args:
            cluster_axis (int, optional): Mesh axis to reduce across. Defaults to all devices.
            subdevice_id (ttnn.SubDeviceId, optional): Worker subdevice.
            num_links (int, optional): Fabric links.
            topology (ttnn.Topology, optional): Fabric topology.

        Returns:
            ttnn.Tensor: ROW_MAJOR WIDTH_SHARDED tensor with the same shape and memory config as the input.
        )doc";

    ttnn::bind_function<"width_sharded_all_reduce", "ttnn.experimental.deepseek.">(
        mod,
        doc,
        &ttnn::experimental::deepseek::width_sharded_all_reduce,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("topology") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek::width_sharded_all_reduce::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_width_sharded_all_reduce(::nanobind::module_& mod) {
    width_sharded_all_reduce::detail::bind_width_sharded_all_reduce(mod);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
