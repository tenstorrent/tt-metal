// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "deepseek_moe_gate_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate {

// Tensor parameters backing the tensor-backed dataflow buffers. create_program_artifacts and
// override_runtime_arguments both use these names.
inline const tt::tt_metal::experimental::TensorParamName kInputTensor{"input_tensor"};
inline const tt::tt_metal::experimental::TensorParamName kBiasTensor{"bias_tensor"};
inline const tt::tt_metal::experimental::TensorParamName kOutputTensor{"output_tensor"};
inline const tt::tt_metal::experimental::TensorParamName kInputIndicesTensor{"input_indices_tensor"};
inline const tt::tt_metal::experimental::TensorParamName kOutputIndicesTensor{"output_indices_tensor"};

// Build the program spec and its run args from current tensors / scalars (cache miss only).
ttnn::device_operation::ProgramArtifacts build_moe_gate_program_artifacts(
    const tensor_args_t& tensor_args, const operation_attributes_t& operation_attrs);

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate
