// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::generic {
namespace nb = nanobind;
// ttnn.freeze_program_hashes (bring-up carry from mstaletovic/combine-lean-sender-llk 1bc1002c13d, binding only).
void bind_freeze_program_hashes(nb::module_& mod);
}  // namespace ttnn::operations::generic
