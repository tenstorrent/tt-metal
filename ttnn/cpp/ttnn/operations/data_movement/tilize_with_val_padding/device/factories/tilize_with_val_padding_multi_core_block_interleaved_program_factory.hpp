// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct TilizeWithValPaddingMultiCoreBlockInterleavedFactory {
    // The plan create_program_artifacts builds from. It depends on the L1 free right now.
    static ttnn::operations::data_movement::BlockPlan block_plan(
        const TilizeWithValPaddingParams& operation_attributes,
        const Tensor& input_tensor,
        const Tensor& output_tensor);

    // The part of that plan free L1 decides, cheap enough for compute_program_hash to key on.
    static ttnn::operations::data_movement::BlockL1Fit block_l1_fit(
        const TilizeWithValPaddingParams& operation_attributes,
        const Tensor& input_tensor,
        const ttnn::Shape& output_padded_shape);

    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TilizeWithValPaddingParams& operation_attributes,
        const Tensor& input_tensor,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
