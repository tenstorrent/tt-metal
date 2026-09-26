// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "hello_world_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct HelloWorldProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const HelloWorldParams& args, const HelloWorldInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::experimental::prim
