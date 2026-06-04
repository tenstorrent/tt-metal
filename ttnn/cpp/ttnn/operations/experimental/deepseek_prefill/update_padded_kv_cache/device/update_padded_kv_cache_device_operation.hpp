// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

struct UpdatePaddedKvCacheDeviceOperation {
    struct operation_attributes_t {
        // Cache slot is linearized as users-outer, layers-inner:
        //   batch_idx = slot_idx * num_layers + layer_idx
        // slot_idx is a per-call device tensor (see tensor_args_t); layer_idx is hashed (structural):
        // it takes only num_layers distinct values, so one cached program per layer is reused across
        // users and chunks -- and a hashed scalar stays correct on the buffer-binding fast cache-hit
        // path (each program bakes its own layer_idx), unlike a non-hashed common rt-arg which would
        // go stale there.
        uint32_t layer_idx;
        uint32_t num_layers;
        uint32_t cluster_axis;
    };

    struct tensor_args_t {
        const Tensor& cache;
        const Tensor& input;
        // Single-element ROW_MAJOR uint32 device tensors, read on-device by the writer kernel.
        // Tensors (not scalar attrs) so their values stay out of the program hash and the
        // buffer-binding fast cache-hit path can patch their addresses -- one cached program (per
        // layer) is reused across users and chunks. NOT hashed by value.
        const Tensor& slot_idx;          // user slot in the batched prefill cache
        const Tensor& kv_actual_global;  // prior valid global KV length in tokens; tile-aligned
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& args,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache

namespace ttnn::prim {

ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const ttnn::Tensor& slot_idx,
    uint32_t layer_idx,
    uint32_t num_layers,
    const ttnn::Tensor& kv_actual_global,
    uint32_t cluster_axis);

}  // namespace ttnn::prim
