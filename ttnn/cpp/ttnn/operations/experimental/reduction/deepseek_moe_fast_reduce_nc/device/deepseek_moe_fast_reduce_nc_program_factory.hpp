// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "deepseek_moe_fast_reduce_nc_device_operation_types.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEFastReduceNCProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DeepseekMoEFastReduceNCParams& operation_attributes,
        const DeepseekMoEFastReduceNCInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
