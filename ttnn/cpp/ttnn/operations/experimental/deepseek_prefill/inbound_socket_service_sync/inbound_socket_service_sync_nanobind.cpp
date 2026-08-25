// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "inbound_socket_service_sync_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "inbound_socket_service_sync.hpp"
#include "ttnn/services/h2d_socket_service.hpp"
#include "ttnn/tensor/d2d_stream_service.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::inbound_socket_service_sync::detail {

void bind_inbound_socket_service_sync(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Wait for the next H2DStreamService transfer, copy it into a device tensor, and
        ack the service core.

        Args:
            service (ttnn.H2DStreamService | ttnn.D2DStreamServiceReceiver): A persistent
                receiver-side service constructed with ``worker_cores`` set (and
                ``metadata_size_bytes`` if used). An H2DStreamService drains a host->device
                transfer; a D2DStreamServiceReceiver drains a device->device transfer.

        Keyword Args:
            metadata_size_bytes (int): When > 0, must match the service's value. Adds a
                second output tensor holding the inline metadata. Default: 0.
            tokens_out (ttnn.Tensor): Caller-owned persistent destination for the tokens.
                When given, the op writes into it and allocates nothing -- required to
                capture this call in a trace, since a trace bakes the destination address
                in at capture time and re-patches nothing on replay. Must match the
                service's per-shard spec, e.g.
                ``ttnn.allocate_tensor_on_device(service.get_per_shard_spec(), mesh_device)``.
                Default: None (allocate a fresh tensor per call).
            metadata_out (ttnn.Tensor): Caller-owned persistent destination for the
                metadata; ``[1, 1, 1, metadata_size_bytes // 4]`` uint32 ROW_MAJOR
                interleaved DRAM. Requires ``metadata_size_bytes > 0``. Default: None.
                Note the framework stamps the tokens' mesh topology onto both outputs,
                so read the metadata per shard (``ttnn.get_device_tensors(md)[k]``)
                rather than composing the whole tensor.

        Returns:
            List[ttnn.Tensor]: ``[tokens]`` when ``metadata_size_bytes == 0``, else
            ``[tokens, metadata]``. Persistent destinations are returned as-is, so the
            caller keeps ownership.
        )doc";

    // Two overloads under one Python name; nanobind dispatches on the `service`
    // arg type (H2DStreamService vs D2DStreamServiceReceiver). The now-overloaded
    // function address must be disambiguated via these typedefs.
    using H2DReceiverFn = std::vector<ttnn::Tensor> (*)(
        const tt::tt_metal::H2DStreamService&,
        uint32_t,
        const std::optional<ttnn::Tensor>&,
        const std::optional<ttnn::Tensor>&);
    using D2DReceiverFn = std::vector<ttnn::Tensor> (*)(
        const ttnn::D2DStreamServiceReceiver&,
        uint32_t,
        const std::optional<ttnn::Tensor>&,
        const std::optional<ttnn::Tensor>&);

    ttnn::bind_function<"inbound_socket_service_sync", "ttnn.experimental.deepseek_prefill.">(
        mod,
        doc,
        ttnn::overload_t(
            static_cast<H2DReceiverFn>(&ttnn::experimental::inbound_socket_service_sync),
            nb::arg("service"),
            nb::kw_only(),
            nb::arg("metadata_size_bytes") = static_cast<uint32_t>(0),
            nb::arg("tokens_out") = std::nullopt,
            nb::arg("metadata_out") = std::nullopt),
        ttnn::overload_t(
            static_cast<D2DReceiverFn>(&ttnn::experimental::inbound_socket_service_sync),
            nb::arg("service"),
            nb::kw_only(),
            nb::arg("metadata_size_bytes") = static_cast<uint32_t>(0),
            nb::arg("tokens_out") = std::nullopt,
            nb::arg("metadata_out") = std::nullopt));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::inbound_socket_service_sync::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_inbound_socket_service_sync(::nanobind::module_& mod) {
    inbound_socket_service_sync::detail::bind_inbound_socket_service_sync(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
