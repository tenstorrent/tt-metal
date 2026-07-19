// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// Handles ROW_MAJOR input: there is no tile padding to strip, so this degenerates to a plain
// unpad/copy - reuses the same reader/writer kernels ttnn::slice already ships for exactly this
// (read/write only the unpadded sticks, skipping padded ones via per-dim counters), since
// output_tensor_end always slices from index 0 (unlike slice's arbitrary start offset).
struct UntilizeWithUnpaddingRowMajorProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output);
};

}  // namespace ttnn::prim
