// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "typecast_device_op_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// Program factory for cross-layout typecast: TILE<->ROW_MAJOR with optional dtype change.
// Uses two-pass compute with intermediate CB:
//   TILE→RM:  typecast(input→intermediate) then untilize(intermediate→output)
//   RM→TILE:  tilize(input→intermediate) then typecast(intermediate→output)
struct TypecastCrossLayoutProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        uint32_t num_cores{};
        uint32_t num_cores_y{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TypecastParams& operation_attributes,
        const TypecastInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::prim
