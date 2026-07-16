// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_codegen_program_factory.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::data_movement {

// Phase 4a translates ops/permute/spec.py's build_permute_rm host section into this descriptor
// (reader_stick_interleaved_unified.cpp SEQ_IDENTITY + writer_permute_rm_interleaved.cpp).
tt::tt_metal::ProgramDescriptor PermuteCodegenDeviceOperation::RowInvariant::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    TT_THROW("PermuteCodegenDeviceOperation::RowInvariant::create_descriptor not yet implemented (phase 4a)");
}

// Phase 4a translates ops/permute/spec.py's build_permute_rm_blocked host section into this
// descriptor (reader_permute_rm_blocked.cpp -> compute_permute_xw_rm.cpp -> writer_permute_rm_blocked.cpp).
tt::tt_metal::ProgramDescriptor PermuteCodegenDeviceOperation::BlockedGeneric::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    TT_THROW("PermuteCodegenDeviceOperation::BlockedGeneric::create_descriptor not yet implemented (phase 4a)");
}

}  // namespace ttnn::operations::data_movement
