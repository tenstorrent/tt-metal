// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::grid_sample {

constexpr uint32_t PRECOMPUTED_GRID_ELEMENTS_PER_POINT = 6;
constexpr uint32_t STANDARD_GRID_ELEMENTS_PER_POINT = 2;

struct GridSample {
    const std::string mode_;
    const std::string padding_mode_;
    const bool use_precomputed_grid_;
    const bool batch_output_channels_;
    const tt::tt_metal::MemoryConfig output_mem_config_;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "mode", "padding_mode", "use_precomputed_grid", "batch_output_channels", "output_mem_config");
    auto attribute_values() const {
        return std::forward_as_tuple(
            this->mode_,
            this->padding_mode_,
            this->use_precomputed_grid_,
            this->batch_output_channels_,
            this->output_mem_config_);
    }
};

}  // namespace ttnn::operations::grid_sample

// Program factory declaration
namespace ttnn::operations::grid_sample {

tt::tt_metal::operation::ProgramWithCallbacks grid_sample_program_factory(
    const Tensor& input_tensor,
    const Tensor& grid_tensor,
    const Tensor& output_tensor,
    const std::string& mode,
    const std::string& padding_mode,
    bool use_precomputed_grid,
    bool batch_output_channels);

}  // namespace ttnn::operations::grid_sample
