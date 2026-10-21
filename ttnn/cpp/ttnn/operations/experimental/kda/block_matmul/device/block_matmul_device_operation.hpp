// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct BlockMatmulParams {
    // One rectangle of worker cores (a set, so the attributes reflect).
    tt::tt_metal::CoreRangeSet cores;
    uint32_t k_block;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct BlockMatmulInputs {
    Tensor a;
    Tensor w;
};

struct BlockMatmulProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const BlockMatmulParams&, const BlockMatmulInputs&, std::vector<Tensor>&);
};

struct BlockMatmulOperation {
    using operation_attributes_t = BlockMatmulParams;
    using tensor_args_t = BlockMatmulInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<BlockMatmulProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return BlockMatmulProgramFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim
