// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "moe_gate_mm_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::moe::moe_gate_mm::program {

struct MoEGateMMProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  cb_s2c_in and cb_s2c_out are
    // sharded onto the input / output tensor buffers and are bound via .buffer
    // for dynamic CB address re-application.  Two semaphores (partial,
    // raw_scores) are declared via SemaphoreDescriptor.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek::moe::moe_gate_mm::program
