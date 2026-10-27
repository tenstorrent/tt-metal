// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dequantize_nvfp4_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {
struct Nvfp4ProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(const Nvfp4Params&, const Nvfp4Inputs&, Tensor&);
};
}  // namespace ttnn::experimental::prim
