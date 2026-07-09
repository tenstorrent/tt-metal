// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::ccl {

namespace nb = nanobind;
void bind_neighbor_pad_conv3d(nb::module_& mod);

// Binds NpConv3dConfig (Conv3dConfig subclass carrying the fused-only controls). Must be called on
// the same module as bind_conv3d and after it, so the Conv3dConfig base type is already registered.
void bind_np_conv3d_config(nb::module_& mod);

}  // namespace ttnn::operations::experimental::ccl
