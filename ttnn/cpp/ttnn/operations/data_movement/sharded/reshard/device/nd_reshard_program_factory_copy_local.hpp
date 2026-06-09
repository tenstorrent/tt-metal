// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "reshard_device_operation_types.hpp"

namespace ttnn::prim {

// Factory for L1<->DRAM or L1->L1 nd reshard (read into local pages in L1)
template <bool local_is_input>
struct NdReshardCopyLocalShardFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ReshardParams& operation_attributes, const ReshardInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
