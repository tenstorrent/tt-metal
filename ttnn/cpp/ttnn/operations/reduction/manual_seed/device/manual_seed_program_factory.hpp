// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "manual_seed_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::reduction::manual_seed::program {
using namespace tt::tt_metal;

struct ManualSeedProgramFactory {
    struct shared_variables_t {};

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void override_runtime_arguments(
        cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

}  // namespace ttnn::operations::reduction::manual_seed::program
