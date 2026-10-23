// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::matmul_decode::detail {

namespace nb = nanobind;
void bind_matmul_decode_operation(nb::module_& mod);
// Descriptor-level bindings (MatmulDecodeParams / MatmulDecodeInputs / MatmulDecodeDeviceOperation
// / per-factory create_descriptor), for models/experimental/ops/descriptors/matmul_decode.py.
// Mirrors ttnn::prim::MatmulDeviceOperation's bindings in matmul_nanobind.cpp.
void bind_matmul_decode_descriptor(nb::module_& mod);

}  // namespace ttnn::operations::experimental::matmul_decode::detail
