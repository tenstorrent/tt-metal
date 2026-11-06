// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::distributed_rms_norm {

namespace nb = nanobind;
void bind_distributed_rms_norm_operation(nb::module_& mod);

}  // namespace ttnn::operations::distributed_rms_norm
