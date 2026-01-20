// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_operation_types.hpp"

#include <tt-metalium/kernel_types.hpp>
#include <ttnn/device_operation.hpp>

namespace ttnn::prim {
// Sharded memory
struct SoftmaxShardedProgramFactoryAttentionOptimized {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernels_id{};
        tt::tt_metal::CBHandle cb_in0_id{}, cb_out0_id{};
        std::optional<tt::tt_metal::CBHandle> cb_in3_id;
        uint32_t num_cores{};
        CoreCoord grid_size;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    static cached_program_t create(const SoftmaxParams&, const SoftmaxInputs&, Tensor&);
    static void override_runtime_arguments(cached_program_t&, const SoftmaxParams&, const SoftmaxInputs&, Tensor&);
};

}  // namespace ttnn::prim
