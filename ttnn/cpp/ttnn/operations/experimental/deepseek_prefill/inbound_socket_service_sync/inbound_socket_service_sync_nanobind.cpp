// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "inbound_socket_service_sync_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "inbound_socket_service_sync.hpp"
#include "ttnn/services/h2d_socket_service.hpp"
#include "ttnn/tensor/d2d_stream_service.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::inbound_socket_service_sync::detail {

void bind_inbound_socket_service_sync(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Wait for the next H2DStreamService transfer, copy it into a freshly-allocated
        device tensor, and ack the service core.

        Args:
            service (ttnn.H2DStreamService | ttnn.D2DStreamServiceReceiver): A persistent
                receiver-side service constructed with ``worker_cores`` set (and
                ``metadata_size_bytes`` if used). An H2DStreamService drains a host->device
                transfer; a D2DStreamServiceReceiver drains a device->device transfer.

        Keyword Args:
            metadata_size_bytes (int): When > 0, must match the service's value. Adds a
                second output tensor holding the inline metadata. Default: 0.

        Returns:
            List[ttnn.Tensor]: ``[tokens]`` when ``metadata_size_bytes == 0``, else
            ``[tokens, metadata]``.
        )doc";

    // Two overloads under one Python name; nanobind dispatches on the `service`
    // arg type (H2DStreamService vs D2DStreamServiceReceiver). The now-overloaded
    // function address must be disambiguated via these typedefs.
    using H2DReceiverFn = std::vector<ttnn::Tensor> (*)(const tt::tt_metal::H2DStreamService&, uint32_t);
    using D2DReceiverFn = std::vector<ttnn::Tensor> (*)(const tt::tt_metal::D2DStreamServiceReceiver&, uint32_t);

    ttnn::bind_function<"inbound_socket_service_sync", "ttnn.experimental.deepseek_prefill.">(
        mod,
        doc,
        ttnn::overload_t(
            static_cast<H2DReceiverFn>(&ttnn::experimental::inbound_socket_service_sync),
            nb::arg("service"),
            nb::kw_only(),
            nb::arg("metadata_size_bytes") = static_cast<uint32_t>(0)),
        ttnn::overload_t(
            static_cast<D2DReceiverFn>(&ttnn::experimental::inbound_socket_service_sync),
            nb::arg("service"),
            nb::kw_only(),
            nb::arg("metadata_size_bytes") = static_cast<uint32_t>(0)));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::inbound_socket_service_sync::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_inbound_socket_service_sync(::nanobind::module_& mod) {
    inbound_socket_service_sync::detail::bind_inbound_socket_service_sync(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
