// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "mhc_split_sinkhorn_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim::qsr {

struct MhcSplitSinkhornProgramFactory {
    using tensor_return_value_t = std::array<Tensor, 3>;

    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MhcSplitSinkhornParams& operation_attributes,
        const MhcSplitSinkhornTensorArgs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::experimental::prim::qsr
