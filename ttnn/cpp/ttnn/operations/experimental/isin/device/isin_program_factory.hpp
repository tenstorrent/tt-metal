// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../isin_common.hpp"

#include "isin_device_op_types.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/isin/isin_cbs.hpp"

namespace ttnn::operations::experimental::isin {

using namespace tt;
using ttnn::device_operation::CachedProgram;
using namespace tt::tt_metal;

struct IsInProgramFactory {
    struct shared_variables_t {
        KernelHandle reader_kernel_id{};
        KernelHandle writer_kernel_id{};
        std::vector<CoreCoord> cores{};
    };

    using cached_program_t = CachedProgram<shared_variables_t>;
    static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);

    static void override_runtime_arguments(
        cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);

    static CBHandle create_cb(
        Program& program,
        const DataType& dtype,
        const IsInCB& is_in_cb,
        const CoreRangeSet& core_range_set,
        const uint32_t& page_size_bytes);

    static KernelHandle create_kernel(
        Program& program,
        const char* kernel_path,
        const CoreRangeSet& core_range_set,
        const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
        const std::vector<uint32_t>& runtime_args = {});
};

}  // namespace ttnn::operations::experimental::isin
