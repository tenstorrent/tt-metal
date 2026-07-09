// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "halo_scatter_device_operation_types.hpp"
#include "halo_scatter_program_factory.hpp"

namespace ttnn::experimental::prim {

struct NpHaloScatterDeviceOperation {
    using operation_attributes_t = NpHaloScatterParams;
    using tensor_args_t = NpHaloScatterInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<NpHaloScatterMeshWorkloadFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Primitive entry point: repack into a newly-allocated padded buffer (interior from interior_src,
// border from the compact halo buffer) and return it.
Tensor halo_scatter(
    const Tensor& compact_buffer,
    const Tensor& interior_src,
    const ttnn::experimental::prim::NpHaloScatterParams& params);

}  // namespace ttnn::prim
