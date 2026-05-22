// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "extract_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::extract {

struct ExtractProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  All CBs are local scratch
    // (no .buffer binding); per-core runtime args carry buffer base addresses
    // and the per-core flat core_id used by the kernel for tile-row sharding.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ExtractParams& operation_attributes, const ExtractInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::extract
