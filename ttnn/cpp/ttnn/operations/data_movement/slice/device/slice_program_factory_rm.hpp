// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

struct SliceRmProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  The src0 CB's total_size /
    // page_size depend on slice_start (via misalignment / unpadded_row_size_bytes)
    // and are intentionally kept out of the program hash (CBDescriptor::total_size
    // is not hashed; page_size is hashed today but the framework applies it on
    // cache hit for forward-compatibility).  apply_descriptor_runtime_args
    // re-runs this factory on the slow-path rebuild and patches the cached
    // program's CB sizing in place — same scheme as the legacy
    // UpdateCircularBufferTotalSize/PageSize path.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
