// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "../untilize_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Untilize's multi-thread path runs one reader thread per compute thread ("lane") and 2 writer threads.
constexpr uint32_t kUntilizeNumLanes = 4;
constexpr uint32_t kUntilizeNumWriterThreads = 2;

// Untilizes interleaved tile input into a row-major interleaved output of the same width with 4 reader,
// 4 compute and 2 writer threads, the data-movement ones with implicit sync. Any height padding must sit
// in the last tile row; its padding rows are dropped. Returns std::nullopt when the tensors do not fit
// this path.
std::optional<ttnn::device_operation::ProgramArtifacts> create_untilize_split_rows_program(
    const Tensor& input, const Tensor& output, bool fp32_dest_acc_en);

struct UntilizeMultiCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const UntilizeOperationAttributes& operation_attributes,
        const UntilizeTensorArgs& tensor_args,
        UntilizeTensorReturnValue& output);
};
}  // namespace ttnn::prim::qsr
