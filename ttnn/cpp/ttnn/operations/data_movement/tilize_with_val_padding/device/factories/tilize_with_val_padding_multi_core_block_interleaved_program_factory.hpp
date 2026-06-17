// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct TilizeWithValPaddingMultiCoreBlockInterleavedFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TilizeWithValPaddingParams& operation_attributes,
        const Tensor& input_tensor,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
