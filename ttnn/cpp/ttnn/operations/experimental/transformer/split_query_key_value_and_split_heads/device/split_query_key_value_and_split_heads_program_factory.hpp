// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "split_query_key_value_and_split_heads_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct SplitFusedQKVAndSplitHeadsProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        uint32_t num_cores_r{};
        uint32_t num_cores_c{};
        uint32_t start_core_x{};
        uint32_t start_core_y{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const SplitQueryKeyValueAndSplitHeadsParams& operation_attributes,
        const SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
        std::vector<Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const SplitQueryKeyValueAndSplitHeadsParams& operation_attributes,
        const SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
        std::vector<Tensor>& output_tensors);
};

}  // namespace ttnn::experimental::prim
