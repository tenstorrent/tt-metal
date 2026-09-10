// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::layer_ack_service {
namespace nb = nanobind;

// The service is exposed entirely as a type (LayerAckService + its methods), so there is no
// py_module() second pass here -- unlike the sibling services, which register free functions in one.
void py_module_types(nb::module_& mod);

}  // namespace ttnn::layer_ack_service
