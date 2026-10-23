// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include <optional>
#include <variant>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

struct DeepseekV4PrefillWriteRowsParams {};

struct DeepseekV4PrefillWriteRowsInputs {
    Tensor input;   // [1, 1, n, W] ROW_MAJOR
    Tensor output;  // [1, 1, R, W] ROW_MAJOR, written in place
    Tensor rows;    // n uint32/int32 ROW_MAJOR (one page): input row i goes to output row rows[i]
};

struct DeepseekV4PrefillWriteRowsOperation {
    using operation_attributes_t = DeepseekV4PrefillWriteRowsParams;
    using tensor_args_t = DeepseekV4PrefillWriteRowsInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& attrs, const tensor_args_t& t, tensor_return_value_t& output);

        // Cache hit: refresh the buffer addresses and the output's row count.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

Tensor deepseek_v4_prefill_write_rows(const Tensor& input, const Tensor& output, const Tensor& rows);

}  // namespace ttnn::prim
