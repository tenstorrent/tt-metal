// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/device_operation.hpp"
#include "pull_all_gather_device_operation_types.hpp"
#include "pull_all_gather_factory.hpp"

namespace ttnn::operations::ccl {

struct PullAllGatherDeviceOperation {
    using operation_attributes_t = PullAllGatherParams;
    using tensor_args_t = PullAllGatherInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    // One path: multicast pull.
    using program_factory_t = std::variant<PullAllGatherFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::ccl
