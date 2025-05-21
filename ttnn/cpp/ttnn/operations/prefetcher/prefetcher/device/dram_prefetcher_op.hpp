// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include <tt-metalium/global_circular_buffer.hpp>


namespace ttnn::operations::dram_prefetcher {

tt::tt_metal::operation::ProgramWithCallbacks dram_prefetcher_multi_core(
    const std::vector<Tensor>& input_tensors,
    const uint32_t num_layers,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb,
    const bool enable_performance_mode = false);

struct DramPrefetcher {
    const std::optional<const GlobalCircularBuffer> global_cb;
    const uint32_t num_layers;
    const bool enable_performance_mode;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::dram_prefetcher
