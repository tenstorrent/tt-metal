// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/global_circular_buffer.hpp>
#include <tuple>

namespace ttnn::prim {

struct DramPrefetcherParams {
    uint32_t num_layers = 0;
    bool enable_performance_mode = false;
    std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;

    static constexpr auto attribute_names = std::forward_as_tuple("num_layers", "enable_performance_mode", "global_cb");
    auto attribute_values() const { return std::forward_as_tuple(num_layers, enable_performance_mode, global_cb); }
};

struct DramPrefetcherInputs {
    std::vector<Tensor> input_tensors;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensors");
    auto attribute_values() const { return std::forward_as_tuple(input_tensors); }
};

}  // namespace ttnn::prim
