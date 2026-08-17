// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "deepseek_moe_gate_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::program {

struct DeepseekMoeGateSharedVariables {
    std::vector<tt::tt_metal::CBHandle> cb_handles;
    std::size_t num_kernel_handles{};
};

struct DeepseekMoeGateProgramFactory {
    using shared_variables_t = DeepseekMoeGateSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        [[maybe_unused]] tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        [[maybe_unused]] tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::program
