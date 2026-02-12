// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct DeepseekMoEFastReduceNCParams {
    uint32_t dim;
    uint64_t split_size;
    tt::tt_metal::MemoryConfig output_memory_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;

    static constexpr auto attribute_names =
        std::forward_as_tuple("dim", "split_size", "output_memory_config", "compute_kernel_config");
    auto attribute_values() const {
        return std::forward_as_tuple(dim, split_size, output_memory_config, compute_kernel_config);
    }
};

struct DeepseekMoEFastReduceNCInputs {
    ttnn::Tensor input_tensor;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensor");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor); }
};

}  // namespace ttnn::experimental::prim
