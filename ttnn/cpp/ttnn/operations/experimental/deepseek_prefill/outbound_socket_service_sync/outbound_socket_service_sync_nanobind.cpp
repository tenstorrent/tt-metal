// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "outbound_socket_service_sync_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "outbound_socket_service_sync.hpp"
// nanobind needs the COMPLETE D2DStreamServiceSender type (typeid / type_caster) to
// bind a function taking it by reference -- the forward declaration in the public
// header is not enough at the binding site.
#include "ttnn/tensor/d2d_stream_service.hpp"
#include "ttnn/services/d2h_socket_service.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::outbound_socket_service_sync::detail {

void bind_outbound_socket_service_sync(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Stage data into an outbound socket service's backing tensor and inc its data_ready
        counter. The inverse of ``inbound_socket_service_sync``. Overload picked by
        ``service`` type.

        NON-BLOCKING: returns once the data is staged and the counter is inc'd.
          * D2DStreamServiceSender (``input`` required): copies ``input`` into the sender
            backing; forwards over fabric once it has ``num_workers`` acks AND the host
            lease (``release_fabric_links()`` / ``wait_for_fabric_links()``).
          * D2HStreamService (``input``/``metadata`` optional, >=1 set): streams to host;
            metadata-only (``input=None``) sends just the record, no payload/lease.

        Args:
            service (ttnn.D2DStreamServiceSender | ttnn.D2HStreamService): persistent
                outbound service (built with worker cores set).

        Keyword Args:
            input (ttnn.Tensor): payload; must match the backing per-shard spec. Default: None.
            metadata (ttnn.Tensor): ``[1, 1, 1, N]`` uint32 blob; size must match the
                service's ``metadata_size_bytes``. Default: None.

        Returns:
            ttnn.Tensor: the (now-filled) service backing tensor.
        )doc";

    ttnn::bind_function<"outbound_socket_service_sync", "ttnn.experimental.deepseek_prefill.">(
        mod,
        doc,
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(
                const ttnn::D2DStreamServiceSender&, const ttnn::Tensor&, const std::optional<ttnn::Tensor>&)>(
                &ttnn::experimental::outbound_socket_service_sync),
            nb::arg("service"),
            nb::arg("input"),
            nb::kw_only(),
            nb::arg("metadata") = std::nullopt),
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(
                const tt::tt_metal::D2HStreamService&,
                const std::optional<ttnn::Tensor>&,
                const std::optional<ttnn::Tensor>&)>(&ttnn::experimental::outbound_socket_service_sync),
            nb::arg("service"),
            nb::kw_only(),
            nb::arg("input") = std::nullopt,
            nb::arg("metadata") = std::nullopt));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::outbound_socket_service_sync::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_outbound_socket_service_sync(::nanobind::module_& mod) {
    outbound_socket_service_sync::detail::bind_outbound_socket_service_sync(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
