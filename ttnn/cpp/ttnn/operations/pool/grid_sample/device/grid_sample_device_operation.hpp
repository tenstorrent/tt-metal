// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/operations/pool/grid_sample/device/grid_sample_device_operation_types.hpp"
#include "ttnn/operations/pool/grid_sample/device/grid_sample_bilinear_program_factory.hpp"
#include "ttnn/operations/pool/grid_sample/device/grid_sample_nearest_program_factory.hpp"

namespace ttnn::operations::pool::grid_sample {

constexpr uint32_t PRECOMPUTED_GRID_ELEMENTS_PER_POINT = 6;
constexpr uint32_t PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST = 2;
constexpr uint32_t STANDARD_GRID_ELEMENTS_PER_POINT = 2;

struct GridSampleOperation {
    using operation_attributes_t = grid_sample::operation_attributes_t;
    using tensor_args_t = grid_sample::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t =
        std::variant<program::GridSampleBilinearProgramFactory, program::GridSampleNearestProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::pool::grid_sample

namespace ttnn::prim {
ttnn::Tensor grid_sample(
    const Tensor& input_tensor,
    const Tensor& grid,
    const std::string& mode = "bilinear",
    const std::string& padding_mode = "zeros",
    bool align_corners = false,
    bool use_precomputed_grid = false,
    bool batch_output_channels = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);
}  // namespace ttnn::prim
