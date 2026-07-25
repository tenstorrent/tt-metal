// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

#include "triangle_solve_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct TriangleSolveProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TriangleSolveParams& operation_attributes,
        const TriangleSolveInputs& tensor_args,
        std::vector<Tensor>& outputs);
};

// Device operation returning a single tensor: the solution X.
struct TriangleSolveDeviceOperation {
    using operation_attributes_t = TriangleSolveParams;
    using tensor_args_t = TriangleSolveInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<TriangleSolveProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Low-level dispatch (used by the public API in ../triangle_solve.cpp).
// Returns {X [1,1,32,32] bf16}.
std::vector<Tensor> triangle_solve(
    const Tensor& l_neg,
    const Tensor& rhs,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
