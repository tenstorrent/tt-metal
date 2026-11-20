// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/gguf/gguf_matmul_device_operation_types.hpp"

namespace ttnn::prim {

struct GgufMatmulSharedVariables {
    std::array<tt::tt_metal::KernelHandle, 2> reader{}, writer{};  // by the core's column parity
    std::vector<CoreCoord> cores;
    bool writer_weight = false;  // decode with a streaming writer: the weight address is its last runtime arg
};

// One row block of the input: the weight's records split evenly over the grid (an N tile's partials added on one core),
// the input multicast in chunks; bands of grid rows split K when it does not fit L1.
struct GgufMatmulProgramFactory {
    using shared_variables_t = GgufMatmulSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const GgufMatmulParams&, const GgufMatmulInputs&, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program, const GgufMatmulParams&, const GgufMatmulInputs&, Tensor& output);
};

// More than one row block of the input: bands of grid rows share M and split N; each core decodes its weight tiles once
// per M chunk into L1 and multiplies them against the multicast rows (two bands split K for few rows).
struct GgufMatmulPrefillProgramFactory {
    using shared_variables_t = GgufMatmulSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const GgufMatmulParams&, const GgufMatmulInputs&, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program, const GgufMatmulParams&, const GgufMatmulInputs&, Tensor& output);
};

// True when GgufMatmulPrefillProgramFactory takes this input (whole row blocks) and weight (its formats, a blocking
// that fits L1).
bool gguf_prefill_supported(const Tensor& input, const Tensor& weight);

// The factories' plans, which depend on the L1 left to programs (L1_SMALL, live L1 buffers), for the program hash:
// the decode's K bands, the prefill's blocking.
std::vector<uint32_t> gguf_decode_plan_key(const Tensor& input, const Tensor& weight);
std::vector<uint32_t> gguf_prefill_plan_key(const Tensor& input, const Tensor& weight);

}  // namespace ttnn::prim
