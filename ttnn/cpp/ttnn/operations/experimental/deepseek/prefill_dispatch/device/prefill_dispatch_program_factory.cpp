// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_dispatch_program_factory.hpp"
#include "prefill_dispatch_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::prefill_dispatch {

PrefillDispatchDeviceOperation::PrefillDispatchProgramFactory::cached_program_t
PrefillDispatchDeviceOperation::PrefillDispatchProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // TODO: Implement program creation
    // - Create program
    // - Add kernels (reader, writer)
    // - Configure CBs
    // - Set runtime args
    // - Return cached program with shared variables

    TT_THROW("PrefillDispatchProgramFactory::create not yet implemented");
}

void PrefillDispatchDeviceOperation::PrefillDispatchProgramFactory::override_runtime_arguments(
    cached_program_t& /*cached_program*/,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // TODO: Implement runtime argument override
    // - Update kernel runtime args for new tensor addresses
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_dispatch
