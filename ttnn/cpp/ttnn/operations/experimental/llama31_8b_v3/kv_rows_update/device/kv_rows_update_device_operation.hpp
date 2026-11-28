// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b_v3 {

struct KvRowsUpdateParams {};

struct KvRowsUpdateInputs {
    Tensor keys;
    Tensor k;
    Tensor values;
    Tensor v;
    std::vector<Tensor> row_positions;
    Tensor page_table;
};

struct KvRowsUpdateProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KvRowsUpdateParams& operation_attributes, const KvRowsUpdateInputs& tensor_args, Tensor& output);
};

struct KvRowsUpdateOperation {
    using operation_attributes_t = KvRowsUpdateParams;
    using tensor_args_t = KvRowsUpdateInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<KvRowsUpdateProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

bool kv_rows_update_supported(
    const Tensor& keys,
    const Tensor& k,
    const Tensor& values,
    const Tensor& v,
    const std::vector<Tensor>& row_positions,
    const Tensor& page_table);

}  // namespace ttnn::experimental::llama31_8b_v3

namespace ttnn::prim {
ttnn::Tensor llama31_8b_v3_kv_rows_update(
    const Tensor& keys,
    const Tensor& k,
    const Tensor& values,
    const Tensor& v,
    const std::vector<Tensor>& row_positions,
    const Tensor& page_table);
}  // namespace ttnn::prim
