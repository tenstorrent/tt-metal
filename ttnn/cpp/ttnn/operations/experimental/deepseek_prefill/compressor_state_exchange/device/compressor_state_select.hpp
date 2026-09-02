// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct CompressorStateSelectParams {
    uint32_t cluster_axis;
};

struct CompressorStateSelectInputs {
    const Tensor& gathered_state;
    const Tensor& initial_state;
};

struct CompressorStateSelectProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const CompressorStateSelectParams&,
        const CompressorStateSelectInputs&,
        Tensor&,
        const std::optional<ttnn::MeshCoordinate>&);
};

struct CompressorStateSelectDeviceOperation {
    using operation_attributes_t = CompressorStateSelectParams;
    using tensor_args_t = CompressorStateSelectInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<CompressorStateSelectProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor compressor_state_select(const Tensor& gathered_state, const Tensor& initial_state, uint32_t cluster_axis);
}  // namespace ttnn::prim
