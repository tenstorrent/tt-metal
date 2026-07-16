// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "to_device_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/experimental/quasar/to_device/to_device.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_to_device(nb::module_& mod) {
    const auto* doc = R"doc(
            Copy tensor from host to device.

            Args:
                tensor (ttnn.Tensor): The tensor to be copied from host to device.
                device (ttnn.Device | ttnn.MeshDevice): The target device where the tensor will be copied.
                memory_config (ttnn.MemoryConfig, optional): The memory configuration to use. Defaults to `None`.

            Kwargs:
                queue_id (ttnn.QueueId, optional): The queue id to use. Defaults to `null`.

            Returns:
                ttnn.Tensor: the tensor copied to device.
        )doc";

    // Bind as a proper ttnn operation object (callable wrapper with the __ttnn_operation__ marker),
    // matching the matmul/reallocate form in this op group, rather than a plain mod.def() free
    // function (which collided with core's "to_device" registration -> AttributeError at import).
    // NOTE: bind_function wraps the free function as a method (adds an implicit `self`), so the
    // keep_alive patient index is shifted by 1 vs the free-function form: device is arg 3 here
    // (0=return, 1=self, 2=tensor, 3=device), so keep_alive<0, 3> keeps the device alive for the
    // lifetime of the returned on-device tensor.
    ttnn::bind_function<"to_device", "ttnn.experimental.quasar.">(
        mod,
        doc,
        nb::overload_cast<
            const ttnn::Tensor&,
            MeshDevice*,
            const std::optional<MemoryConfig>&,
            std::optional<ttnn::QueueId>>(&ttnn::operations::experimental::quasar::to_device),
        nb::arg("tensor"),
        nb::arg("device"),
        nb::arg("memory_config") = nb::none(),
        nb::kw_only(),
        nb::arg("queue_id") = nb::none(),
        nb::keep_alive<0, 3>());
}

}  // namespace ttnn::operations::experimental::quasar::detail
