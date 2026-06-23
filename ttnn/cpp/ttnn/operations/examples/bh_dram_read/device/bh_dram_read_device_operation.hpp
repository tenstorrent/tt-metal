// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::examples {

struct BhDramReadDeviceOperation {
    // No tunable attributes for this minimal op.
    struct operation_attributes_t {};

    struct tensor_args_t {
        const Tensor& input_tensor;
    };

    // Read-only op: the "output" is the input tensor aliased unchanged.
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Single descriptor-based factory: one worker core per DRAM bank.
    struct DramBankCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<DramBankCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::examples

namespace ttnn::prim {
ttnn::operations::examples::BhDramReadDeviceOperation::tensor_return_value_t bh_dram_read(const Tensor& input_tensor);
}  // namespace ttnn::prim
