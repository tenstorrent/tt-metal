// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct ConvolutionHistoryRowsParams {
    uint32_t width;
    uint32_t column_offset;
    uint32_t sequence_parallel_axis;
};

struct ConvolutionHistoryRowsInputs {
    Tensor input;
    Tensor actual_start;
    std::optional<Tensor> actual_end;
};

struct ConvolutionHistoryRowsProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const ConvolutionHistoryRowsParams&,
        const ConvolutionHistoryRowsInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

struct ConvolutionHistoryRowsOperation {
    using operation_attributes_t = ConvolutionHistoryRowsParams;
    using tensor_args_t = ConvolutionHistoryRowsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ConvolutionHistoryRowsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor convolution_history_rows(
    const Tensor& input,
    const Tensor& actual_start,
    uint32_t width,
    uint32_t column_offset,
    const std::optional<Tensor>& actual_end,
    uint32_t sequence_parallel_axis);

}  // namespace ttnn::experimental::prim
