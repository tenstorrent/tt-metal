// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "fill_rm_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "fill_rm_device_operation_types.hpp"

namespace ttnn::operations::data_movement::fill_rm {

struct FillRMDeviceOperation {
    using operation_attributes_t = FillRmParams;
    using tensor_args_t = FillRmInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::FillRMProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        const tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::data_movement::fill_rm

namespace ttnn::prim {
ttnn::Tensor fill_rm(
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    uint32_t hFill,
    uint32_t wFill,
    const Tensor& input,
    float val_hi,
    float val_lo,
    const MemoryConfig& output_memory_config);
}  // namespace ttnn::prim
