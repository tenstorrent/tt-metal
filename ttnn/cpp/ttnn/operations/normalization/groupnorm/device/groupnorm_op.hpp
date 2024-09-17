// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::constants;

namespace ttnn::operations::normalization {

/**
Ref: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
>>> input = torch.randn(20, 6, 10, 10)
>>> # Separate 6 channels into 3 groups
>>> m = nn.GroupNorm(3, 6)
>>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
>>> m = nn.GroupNorm(6, 6)
>>> # Put all 6 channels into a single group (equivalent with LayerNorm)
>>> m = nn.GroupNorm(1, 6)
>>> # Activating the module
>>> output = m(input)
*/

struct GroupNormShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    MathFidelity math_fidelity;
    DataType im_data_format;
    DataType out_data_format;
    bool inplace;
    Layout output_layout;
};

operation::ProgramWithCallbacks groupnorm_multi_core_sharded(
    const Tensor &a,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    const std::optional<const Tensor> input_mask,
    Tensor& output,
    float eps,
    const uint32_t num_groups,
    const uint32_t num_batches,
    MathFidelity fidelity,
    DataType im_data_format,
    CoreCoord grid_size,
    bool inplace
);

struct GroupNorm {
    float eps;
    uint32_t num_groups;
    MemoryConfig output_mem_config;
    GroupNormShardedMultiCoreProgramConfig program_config;

    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
};


}   // namespace operations::normalization
