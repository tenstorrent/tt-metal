// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "group_attn_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct GroupAttnMatmulProgramFactory {
    // Metal 2.0 factory (MetalV2FactoryConcept): returns a ProgramSpec + its ProgramRunArgs.
    // Several DFB (ex-CB) sizes depend on (KV_HEADS, Mt, Kt, Nt) computed from the input shapes.
    // The op has no custom compute_program_hash; the default device-operation hash already keys on
    // tensor specs / padded_shape, so each unique DFB sizing lands in its own program-cache entry.
    // On cache hit the framework re-patches only the tensor bindings (the TensorAccessor base
    // addresses and the borrowed-DFB backing L1 addresses); DFB entry_size/num_entries are not
    // re-applied (the cached spec already carries the correct values).
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const GroupAttnMatmulParams& operation_attributes,
        const GroupAttnMatmulInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
