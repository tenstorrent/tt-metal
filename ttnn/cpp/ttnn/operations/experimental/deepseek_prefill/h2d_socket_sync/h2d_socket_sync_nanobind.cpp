// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "h2d_socket_sync_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "h2d_socket_sync.hpp"
#include "ttnn/services/h2d_socket_service.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::h2d_socket_sync::detail {

void bind_h2d_socket_sync(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Wait for the next H2DStreamService transfer, copy it into a freshly-allocated
        device tensor, and ack the service core.

        Args:
            service (ttnn.H2DStreamService): A persistent service constructed with
                ``worker_cores`` set (and ``metadata_size_bytes`` if used).

        Keyword Args:
            metadata_size_bytes (int): When > 0, must match the service's value. Adds a
                second output tensor holding the inline metadata. Default: 0.

        Returns:
            List[ttnn.Tensor]: ``[tokens]`` when ``metadata_size_bytes == 0``, else
            ``[tokens, metadata]``.
        )doc";

    ttnn::bind_function<"h2d_socket_sync", "ttnn.experimental.deepseek_prefill.">(
        mod,
        doc,
        &ttnn::experimental::h2d_socket_sync,
        nb::arg("service"),
        nb::kw_only(),
        nb::arg("metadata_size_bytes") = static_cast<uint32_t>(0));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::h2d_socket_sync::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_h2d_socket_sync(::nanobind::module_& mod) { h2d_socket_sync::detail::bind_h2d_socket_sync(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
