// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "convert_to_chw_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <optional>

#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct ConvertToCHWProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ConvertToCHWParams& operation_attributes, const Tensor& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const ConvertToCHWParams& operation_attributes,
        const Tensor& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
