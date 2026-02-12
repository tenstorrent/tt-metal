// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct UntilizeWithUnpaddingMultiCoreShardedProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::CBHandle cb_src0{};
        tt::tt_metal::CBHandle cb_sharded_output{};
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const UntilizeWithUnpaddingParams& operation_attributes,
        const Tensor& input,
        const Tensor& output);
};

}  // namespace ttnn::prim
