// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::transformer {

namespace nb = nanobind;
// Private, host-only introspection of op-selected recipe blocking (tests and tooling).
void bind_sdpa_recipe_blocking(nb::module_& mod);
}  // namespace ttnn::operations::transformer
