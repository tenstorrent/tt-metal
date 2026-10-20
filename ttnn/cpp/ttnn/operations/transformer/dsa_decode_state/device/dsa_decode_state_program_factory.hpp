// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "dsa_decode_state_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct DsaDecodeStateProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DsaDecodeStateParams& attrs, const DsaDecodeStateInputs& in, std::vector<Tensor>& outputs);
};

}  // namespace ttnn::prim
