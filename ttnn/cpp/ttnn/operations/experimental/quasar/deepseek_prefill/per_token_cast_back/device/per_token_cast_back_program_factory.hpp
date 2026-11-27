// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload/legacy/program_descriptors.hpp>

#include "per_token_cast_back_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim::qsr::per_token_cast_back {

// Cache hits: every buffer address (input_e4m3, input_scale, output and, on the token-count-aware path, the three
// expert_* metadata tensors) reaches the kernels as a Buffer* binding, which the framework re-points on a hit.
// Everything else in the runtime args (core index, H, per-core row split) derives from hashed tensor specs and
// attributes, so no override_runtime_arguments is needed.
struct PerTokenCastBackProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PerTokenCastBackParams& operation_attributes,
        const PerTokenCastBackInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim::qsr::per_token_cast_back
