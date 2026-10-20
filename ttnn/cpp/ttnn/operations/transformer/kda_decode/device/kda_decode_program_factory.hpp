// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "kda_decode_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct KdaDecodeProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KdaDecodeParams& attrs, const KdaDecodeInputs& in, std::vector<Tensor>& outputs);
};

}  // namespace ttnn::prim
