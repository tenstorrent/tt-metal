// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_codegen_device_operation.hpp"
#include "permute_codegen_program_factory.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::data_movement {

// Placeholder: builds the reader_stick_interleaved_unified (SEQ_IDENTITY) + writer_permute_rm_interleaved
// kernel pair (see permute.yaml). Phase 4a fills in the real descriptor.
tt::tt_metal::ProgramDescriptor PermuteCodegenDeviceOperation::MultiCoreRowInvariant::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    TT_THROW("PermuteCodegen: MultiCoreRowInvariant::create_descriptor is not implemented yet");
}

// Placeholder: builds the reader_permute_rm_blocked -> compute_permute_xw_rm -> writer_permute_rm_blocked
// kernel chain (see permute.yaml). Phase 4a fills in the real descriptor.
tt::tt_metal::ProgramDescriptor PermuteCodegenDeviceOperation::MultiCoreBlockedGeneric::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    TT_THROW("PermuteCodegen: MultiCoreBlockedGeneric::create_descriptor is not implemented yet");
}

}  // namespace ttnn::operations::data_movement
