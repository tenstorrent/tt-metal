// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_codegen_program_factory.hpp"

#include <tt-metalium/assert.hpp>

#include "gather_codegen_device_operation.hpp"

namespace ttnn::prim {

// Phase 4a fills in the real translation from BuildContext-derived cache_key_fields
// (Ht/Wt_input/Wt_index/index_valid_h_last/index_valid_w_last/index_ht_per_batch) into
// CBDescriptors/KernelDescriptors over kernels/gather_reader.cpp + gather_writer.cpp.
tt::tt_metal::ProgramDescriptor GatherCodegenProgramFactoryInterleaved::create_descriptor(
    const GatherCodegenParams&, const GatherCodegenInputs&, Tensor&) {
    TT_THROW("GatherCodegenProgramFactoryInterleaved::create_descriptor is not yet implemented");
}

tt::tt_metal::ProgramDescriptor GatherCodegenProgramFactoryTiled::create_descriptor(
    const GatherCodegenParams&, const GatherCodegenInputs&, Tensor&) {
    TT_THROW("GatherCodegenProgramFactoryTiled::create_descriptor is not yet implemented");
}

tt::tt_metal::ProgramDescriptor GatherCodegenProgramFactoryStreaming::create_descriptor(
    const GatherCodegenParams&, const GatherCodegenInputs&, Tensor&) {
    TT_THROW("GatherCodegenProgramFactoryStreaming::create_descriptor is not yet implemented");
}

}  // namespace ttnn::prim
