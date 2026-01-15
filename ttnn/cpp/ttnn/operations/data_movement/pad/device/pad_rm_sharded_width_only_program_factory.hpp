// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>

#include "ttnn/device_operation.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim {

struct PadRmShardedWidthOnlySharedVariables {
    tt::tt_metal::CBHandle input_shard_cb{};
    tt::tt_metal::CBHandle output_shard_cb{};
};

struct PadRmShardedWidthOnlyProgramFactory {
    using shared_variables_t = PadRmShardedWidthOnlySharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const PadParams& operation_attributes,
        const PadInputs& tensor_args,
        Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
