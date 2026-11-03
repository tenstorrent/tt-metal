// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dequantize_mxfp8_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {
struct Mxfp8ProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(const Mxfp8Params&, const Mxfp8Inputs&, Tensor&);
};
}  // namespace ttnn::experimental::prim
