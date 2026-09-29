// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "flat_routed_expert_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert {

// Buffers whose addresses the runtime args carry (re-applied on every program-cache hit).
enum class AddrSrc : uint8_t {
    X,
    Y,
    Counts,
    Regions,
    Ids,
    GateUp,
    Down,
    ReaderDown,
    Done,
    Arena,
    Words,
    TokenIndex,
    Count
};

struct AddrPatch {
    tt::tt_metal::KernelHandle kernel;
    tt::tt_metal::CoreCoord core;
    uint32_t index;
    AddrSrc src;
    uint32_t offset;  // byte offset added to the buffer address
};

struct FlatRoutedExpertSharedVariables {
    std::vector<AddrPatch> patches;
    std::vector<std::pair<tt::tt_metal::CBHandle, uint32_t>> arena_cbs;  // (CB, arena byte offset)
};

struct FlatRoutedExpertProgramFactory {
    using shared_variables_t = FlatRoutedExpertSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FlatRoutedExpertParams& operation_attributes,
        const FlatRoutedExpertInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const FlatRoutedExpertParams& operation_attributes,
        const FlatRoutedExpertInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert
