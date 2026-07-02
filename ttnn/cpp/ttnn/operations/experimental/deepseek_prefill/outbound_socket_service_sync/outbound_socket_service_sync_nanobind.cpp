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
        Copy ``input`` into a D2DStreamServiceSender's backing tensor and signal the
        service to forward it over fabric. The inverse of ``inbound_socket_service_sync``.

        NON-BLOCKING: returns once the data is staged and the service's data_ready
        counter is inc'd. The sender forwards once it has ``num_workers`` acks AND the
        host grants the fabric lease (``service.release_fabric_links()``). Drive
        back-pressure with ``service.wait_for_fabric_links()`` before the next call.

        Args:
            service (ttnn.D2DStreamServiceSender): persistent sender built with
                ``sender_worker_cores`` set.
            input (ttnn.Tensor): the producing stage's output. Must share the sender
                backing tensor's per-shard spec.

        Keyword Args:
            metadata (ttnn.Tensor): optional ``[1, 1, 1, N]`` uint32 blob forwarded
                inline to the sender service core. Its size must match the service's
                ``metadata_size_bytes``. Default: None.

        Returns:
            ttnn.Tensor: the (now-filled) sender backing tensor.
        )doc";

    ttnn::bind_function<"outbound_socket_service_sync", "ttnn.experimental.deepseek_prefill.">(
        mod,
        doc,
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(
                const tt::tt_metal::D2DStreamServiceSender&, const ttnn::Tensor&, const std::optional<ttnn::Tensor>&)>(
                &ttnn::experimental::outbound_socket_service_sync),
            nb::arg("service"),
            nb::arg("input"),
            nb::kw_only(),
            nb::arg("metadata") = std::nullopt),
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(const tt::tt_metal::D2HStreamService&, const ttnn::Tensor&)>(
                &ttnn::experimental::outbound_socket_service_sync),
            nb::arg("service"),
            nb::arg("record")));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::outbound_socket_service_sync::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_outbound_socket_service_sync(::nanobind::module_& mod) {
    outbound_socket_service_sync::detail::bind_outbound_socket_service_sync(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
