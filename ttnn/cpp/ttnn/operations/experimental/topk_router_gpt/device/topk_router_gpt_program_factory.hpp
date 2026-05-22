// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "topk_router_gpt_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::topk_router_gpt::program {

struct TopkRouterGptProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::topk_router_gpt::program
