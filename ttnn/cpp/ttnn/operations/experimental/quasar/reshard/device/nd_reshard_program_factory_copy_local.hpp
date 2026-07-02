// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "reshard_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Factory for L1<->DRAM or L1->L1 nd reshard (read into local pages in L1)
template <bool local_is_input>
struct NdReshardCopyLocalShardFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshardParams& operation_attributes, const ReshardInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim::qsr
