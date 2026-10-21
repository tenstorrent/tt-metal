// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::kda::ring_relay_prefix::detail {

namespace nb = nanobind;
void bind_ring_relay_prefix(nb::module_&);

}  // namespace ttnn::operations::experimental::kda::ring_relay_prefix::detail
