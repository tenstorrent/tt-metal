// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

#include "gated_delta_prefill_query_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct GatedDeltaPrefillQueryProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GatedDeltaPrefillQueryParams& operation_attributes,
        const GatedDeltaPrefillQueryInputs& tensor_args,
        std::vector<Tensor>& outputs);
};

// Device operation returning two tensors: {output token O, updated state S'}.
struct GatedDeltaPrefillQueryDeviceOperation {
    using operation_attributes_t = GatedDeltaPrefillQueryParams;
    using tensor_args_t = GatedDeltaPrefillQueryInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<GatedDeltaPrefillQueryProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Low-level dispatch (used by the public API in ../gated_delta_prefill_query.cpp).
// Returns {O [1,1,Nv,d] bf16, state' [1,Nv,d,d] fp32}.
std::vector<Tensor> gated_delta_prefill_query(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& gate,
    const Tensor& decay,
    const Tensor& state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
