// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct CacheBundleAllocationParams {
    uint32_t slot_id;
    uint32_t actual_start;
    uint32_t actual_end;
    uint32_t page_size;
};
struct CacheBundleAllocationInputs {
    Tensor page_table;
    Tensor allocated_pages;
    Tensor free_list;
    Tensor free_count;
};
struct CacheBundleAllocationProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const CacheBundleAllocationParams&, const CacheBundleAllocationInputs&, Tensor&);
    static void override_runtime_arguments(
        tt::tt_metal::Program&,
        const CacheBundleAllocationParams&,
        const CacheBundleAllocationInputs&,
        Tensor&,
        const std::optional<ttnn::MeshCoordinate>& = std::nullopt);
};
struct UpdateCacheBundleAllocationDeviceOperation {
    using operation_attributes_t = CacheBundleAllocationParams;
    using tensor_args_t = CacheBundleAllocationInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<CacheBundleAllocationProgramFactory>;
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static Tensor create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::experimental::prim
