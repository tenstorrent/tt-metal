// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::inbound_socket_service_sync::detail {
namespace nb = nanobind;
void bind_inbound_socket_service_sync(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::inbound_socket_service_sync::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_inbound_socket_service_sync(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
