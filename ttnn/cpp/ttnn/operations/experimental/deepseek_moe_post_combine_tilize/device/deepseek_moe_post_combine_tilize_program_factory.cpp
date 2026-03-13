// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/deepseek_moe_post_combine_tilize_program_factory.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

DeepseekMoEPostCombineTilizeProgramFactory::cached_program_t DeepseekMoEPostCombineTilizeProgramFactory::create(
    const DeepseekMoEPostCombineTilizeParams&, const DeepseekMoEPostCombineTilizeInputs&, ttnn::Tensor&) {
    auto program = Program();

    return cached_program_t{std::move(program), {5}};
}

void DeepseekMoEPostCombineTilizeProgramFactory::override_runtime_arguments(
    cached_program_t&,
    const DeepseekMoEPostCombineTilizeParams&,
    const DeepseekMoEPostCombineTilizeInputs&,
    ttnn::Tensor&) {
    // TODO: (GR)
}

}  // namespace ttnn::experimental::prim
