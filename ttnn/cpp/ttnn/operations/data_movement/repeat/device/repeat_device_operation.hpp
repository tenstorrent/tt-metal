// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
namespace ttnn {

struct RepeatDeviceOperation {
    const uint32_t m_num_repeats;
    const bool m_is_last_dim;
    tt::tt_metal::MemoryConfig m_output_mem_config;

    // Required functions to all tensor op functions
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};
}  // namespace ttnn
