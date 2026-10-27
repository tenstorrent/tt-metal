// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "extract_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::extract {

struct ExtractProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ExtractParams& operation_attributes, const ExtractInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::extract
