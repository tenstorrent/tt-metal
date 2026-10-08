// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "simple_add_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

#include <variant>

namespace ttnn::prim::qsr {

// C = A + B on a single node: on a Quasar Neo cluster the reader runs on 4 DM cores, the writer on 2 and the
// compute kernel on all 4 Tensix engines (each a single thread on Wormhole/Blackhole). A, B and C are bfloat16,
// TILE layout, DRAM interleaved, with the same shape.
struct SimpleAddDeviceOperation {
    using operation_attributes_t = SimpleAddParams;
    using tensor_args_t = SimpleAddInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SingleNodeProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<SingleNodeProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

ttnn::Tensor simple_add(
    const ttnn::Tensor& input_a, const ttnn::Tensor& input_b, const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim::qsr
