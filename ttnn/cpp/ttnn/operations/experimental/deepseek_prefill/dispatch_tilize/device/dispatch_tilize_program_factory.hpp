// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dispatch_tilize_types.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize {

struct DispatchTilizeProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DispatchTilizeParams& operation_attributes,
        const DispatchTilizeInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize
