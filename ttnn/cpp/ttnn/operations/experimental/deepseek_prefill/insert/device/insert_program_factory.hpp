// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "insert_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::insert {

struct InsertProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const InsertParams& operation_attributes, const InsertInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::insert
