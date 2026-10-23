// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "hybrid_routed_expert_ffn_types.hpp"
#include "hybrid_overlap_program_factory.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include <tt-metalium/device.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

// ONE device operation carrying both routed-expert implementations, so a layer dispatches once.
//
// This is not two ops behind one entry point: create_descriptor builds both halves into a single
// ProgramDescriptor with three merged kernels, one per RISC-V. A program holds at most one kernel
// per processor per core and both halves want the same 88, so side-by-side placement is rejected
// by the framework outright -- the halves have to share binaries.
// Bytes of L1 scratch per core the op needs when the fused pass runs. Exposed because the caller has to
// allocate the arena: the program keeps a raw pointer into it, so it must outlive the program,
// which a buffer made inside create_descriptor would not.
uint32_t hybrid_l1_arena_bytes(tt::tt_metal::IDevice* device);

struct HybridRoutedExpertFfnDeviceOperation {
    using operation_attributes_t = HybridRoutedExpertFfnParams;
    using tensor_args_t = HybridRoutedExpertFfnInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<HybridSoloProgramFactory, HybridOverlapProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
