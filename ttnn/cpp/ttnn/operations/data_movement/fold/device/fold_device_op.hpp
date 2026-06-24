// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::data_movement {

// Returns true if the user's requested output config (or std::nullopt) is satisfiable by the
// zero-NOC `MultiCore` fast path. Shared between the composite layer and `prim::fold` so the
// gating predicate stays in one place. Mirrors the `is_native_*` predicates in transpose/repeat.
bool override_compatible_with_fast_path(const std::optional<tt::tt_metal::MemoryConfig>& override_mc);

struct Fold {
    struct operation_attributes_t {
        uint32_t stride_h{};
        uint32_t stride_w{};
        // Gates the zero-NOC `MultiCore` fast path; everything else routes to MultiCoreDRAMFold.
        bool is_height_sharded_rm_fast_path{};
        // User-requested output memory config (mirrors `repeat`/`transpose`); std::nullopt → derive from input.
        std::optional<tt::tt_metal::MemoryConfig> output_memory_config{};
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct MultiCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    struct MultiCoreDRAMFold {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<MultiCore, MultiCoreDRAMFold>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::Fold::tensor_return_value_t fold(
    const ttnn::Tensor& input_tensor,
    uint32_t stride_h,
    uint32_t stride_w,
    bool is_height_sharded_rm_fast_path,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);
}  // namespace ttnn::prim
