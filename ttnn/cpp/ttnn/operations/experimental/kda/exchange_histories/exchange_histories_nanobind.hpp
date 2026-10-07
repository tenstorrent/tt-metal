// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::kda::exchange_histories::detail {

namespace nb = nanobind;
void bind_exchange_histories(nb::module_&);

}  // namespace ttnn::operations::experimental::kda::exchange_histories::detail
