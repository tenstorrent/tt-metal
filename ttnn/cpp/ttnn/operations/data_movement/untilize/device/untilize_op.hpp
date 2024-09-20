// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement {

namespace untilize_helpers {

uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks);

}  // namespace untilize_helpers

struct Untilize {
    const MemoryConfig output_mem_config;
    const bool use_multicore;
    const bool use_pack_untilize;
    const bool fp32_dest_acc_en;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::data_movement
