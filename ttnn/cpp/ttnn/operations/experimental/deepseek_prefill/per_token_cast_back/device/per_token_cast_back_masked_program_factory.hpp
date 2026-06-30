// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "per_token_cast_back_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim::per_token_cast_back {

// Contract-2 (ProgramDescriptor / WorkloadDescriptor) program factory for the masked decompress
// path. Unlike the plain factory it builds one ProgramDescriptor per mesh coordinate, because each
// device's valid-row window (counter_offset) is derived from its linearized mesh coordinate.
struct MaskedPerTokenCastBackProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const PerTokenCastBackParams& operation_attributes,
        const PerTokenCastBackInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim::per_token_cast_back
