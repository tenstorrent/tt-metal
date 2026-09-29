// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

#include "moe_ag_local_reduce_device_operation_types.hpp"

namespace ttnn::prim {

// All-gather MoE local reduce: partial[g] = sum over this chip's local experts of w[g, k] * y[y_slot[g, k]] for the
// column's tokens (fp32 DEST, the packer accumulating the pairs in L1), over the whole worker grid.
struct MoeAgLocalReduceDeviceOperation {
    using operation_attributes_t = MoeAgLocalReduceParams;
    using tensor_args_t = MoeAgLocalReduceInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

inline constexpr uint32_t MOE_AG_LOCAL_REDUCE_MAX_K = 32;  // weights staged in 64 B L1 pages

// Returns [own] (phase 0 split: [own, other]; phase 1: [other]; phase 2: [own]).
std::vector<Tensor> moe_ag_local_reduce(
    const Tensor& y,
    const Tensor& y_slot,
    const Tensor& weights,
    const Tensor& chip_info,
    uint32_t chunk_size_per_chip,
    uint32_t phase = 0,
    bool split = false,
    bool tiled = false,
    const std::optional<Tensor>& peer = std::nullopt,
    const std::vector<Tensor>& preallocated_outputs = {});

}  // namespace ttnn::prim
