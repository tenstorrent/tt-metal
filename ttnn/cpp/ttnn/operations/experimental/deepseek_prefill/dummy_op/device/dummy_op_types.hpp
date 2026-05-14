// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dummy_op {

struct DummyOpParams {
    // Baked into the kernel as a compile-time constant. Changing it triggers a
    // program-cache miss / recompile.
    uint32_t num_iter;

    // CoreRangeSet for kernel placement. Must consist of exactly one CoreRange
    // spanning exactly one Tensix row (start.y == end.y).
    CoreRangeSet worker_core_range_set;

    static constexpr auto attribute_names = std::forward_as_tuple("num_iter", "worker_core_range_set");

    auto attribute_values() const { return std::forward_as_tuple(num_iter, worker_core_range_set); }
};

struct DummyOpInputs {
    Tensor input_tensor;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dummy_op
