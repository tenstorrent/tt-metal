// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::binary_practice {

struct BinaryPracticeQuasarParams {
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct BinaryPracticeQuasarInputs {
    Tensor a;
    Tensor b;
};

// Metal 2.0 factory: one ProgramSpec with a reader, a compute and a writer kernel on every node that has work.
struct BinaryPracticeQuasarProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const BinaryPracticeQuasarParams& params, const BinaryPracticeQuasarInputs& inputs, Tensor& output);
};

struct BinaryPracticeQuasarDeviceOperation {
    using operation_attributes_t = BinaryPracticeQuasarParams;
    using tensor_args_t = BinaryPracticeQuasarInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<BinaryPracticeQuasarProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::binary_practice

namespace ttnn::prim {
Tensor binary_practice_quasar(const Tensor& a, const Tensor& b);
}  // namespace ttnn::prim
