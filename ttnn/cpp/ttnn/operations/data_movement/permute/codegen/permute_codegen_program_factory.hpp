// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

inline constexpr uint32_t kPermuteCodegenMaxDims = 8;

struct PermuteCodegenParams {
    // Zero-padded past the input rank.
    std::array<uint32_t, kPermuteCodegenMaxDims> dims{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct PermuteCodegenInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
};

struct PermuteCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PermuteCodegenParams& operation_attributes,
        const PermuteCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
