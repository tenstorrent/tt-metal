// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::pcie_p2p {

namespace nb = nanobind;
void bind_pcie_p2p_api(nb::module_& mod);

}  // namespace ttnn::pcie_p2p
