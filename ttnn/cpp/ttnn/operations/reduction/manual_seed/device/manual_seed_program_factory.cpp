// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::reduction::manual_seed::program {
using namespace tt::tt_metal;

ManualSeedProgramFactory::cached_program_t ManualSeedProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*output_tensor*/) {
    tt::tt_metal::Program program{};

    return cached_program_t{std::move(program), {}};
}

void ManualSeedProgramFactory::override_runtime_arguments(
    cached_program_t& /*cached_program*/,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*output_tensor*/) {
    // TODO: Fill
}
}  // namespace ttnn::operations::reduction::manual_seed::program
