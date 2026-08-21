// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/api/ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/sliding_window/halo/device/halo_device_operation_types.hpp"

namespace ttnn::prim {

struct UntilizeWithHaloProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const HaloParams& operation_attributes, const Tensor& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
