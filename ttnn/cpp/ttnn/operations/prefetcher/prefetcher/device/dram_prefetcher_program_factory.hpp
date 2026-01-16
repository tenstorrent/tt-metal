// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "dram_prefetcher_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::dram_prefetcher::program {

struct DramPrefetcherProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::CBHandle tensor_addrs_cb{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const DramPrefetcherParams& operation_attributes,
        const DramPrefetcherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const DramPrefetcherParams& operation_attributes,
        const DramPrefetcherInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::operations::dram_prefetcher::program
