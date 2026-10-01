// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::pcie_exchange::detail {

namespace nb = nanobind;

void bind_experimental_pcie_exchange_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::pcie_exchange::detail
